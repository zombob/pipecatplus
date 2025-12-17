"""pipecat.services.ali.stt 的 Docstring."""

# funasr/stt.py  -- 基于 WebSocket 的 FunASR STT，无 dashscope 依赖
import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Optional

import websockets

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    StopFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601


class AliSTTService(STTService):
    """阿里云 WebSocket 版 STTService."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "fun-asr-realtime",
        ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/",
        sample_rate: int = 16000,
        sample_format: str = "pcm",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.api_key = api_key or os.environ.get("FUNASR_API_KEY")
        if not self.api_key:
            raise ValueError("AliSTTService requires api_key")

        self.model = model
        self.ws_url = ws_url
        self._sample_rate = sample_rate
        self._sample_format = sample_format

        # websocket 对象
        self.ws = None
        self.task_id = None

        # 控制标志
        self.running = False
        self.receiver_task = None

        # 当前用户
        self._user_id = "default"

    # ------------------------------------------------------
    # WebSocket 连接 + run-task
    # ------------------------------------------------------
    async def _connect(self):
        self.task_id = uuid.uuid4().hex[:32]

        # 将 additional_headers 改为 extra_headers 以修复 create_connection 参数错误
        # 请确保这里的 self._headers 与您代码中定义的 headers 变量名一致
        self.ws = await websockets.connect(
            self.ws_url, extra_headers={"Authorization": f"bearer {self.api_key}"}
        )

        # 发送 run-task
        run_task_msg = {
            "header": {"action": "run-task", "task_id": self.task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": self.model,
                "parameters": {"sample_rate": self._sample_rate, "format": self._sample_format},
                "input": {},
            },
        }
        await self.ws.send(json.dumps(run_task_msg))

        # 等待 task-started
        while True:
            msg = json.loads(await self.ws.recv())
            if msg["header"]["event"] == "task-started":
                print("FunASR WebSocket STT started.")
                break

    # ------------------------------------------------------
    # WebSocket 接收线程
    # ------------------------------------------------------
    async def _receiver_loop(self):
        try:
            while self.running:
                raw = await self.ws.recv()

                # 二进制包不会进入此逻辑（只接收文本 JSON）
                if isinstance(raw, bytes):
                    continue

                msg = json.loads(raw)
                event = msg["header"]["event"]

                # 处理识别结果
                if event == "result-generated":
                    sentence = msg["payload"]["output"]["sentence"]
                    text = sentence.get("text", "")

                    if not text:
                        continue

                    # 句末标志
                    is_final = sentence.get("sentence_end", False)

                    frame = (
                        TranscriptionFrame(
                            text, self._user_id, time_now_iso8601(), language=None, result=sentence
                        )
                        if is_final
                        else InterimTranscriptionFrame(
                            text, self._user_id, time_now_iso8601(), language=None, result=sentence
                        )
                    )

                    # 输出到 Pipecat pipeline
                    await self.push_frame(frame)

                elif event == "task-finished":
                    print("FunASR task finished.")
                    break

                elif event == "task-failed":
                    print("FunASR task-failed message:", msg)
                    break

        except Exception as e:
            print("Receiver loop exception:", e)

    # ------------------------------------------------------
    # Pipecat 生命周期：Start
    # ------------------------------------------------------
    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._user_id = frame.metadata.get("user_id", "default")
        self.running = True

        await self._connect()

        # 启动接收任务
        loop = asyncio.get_running_loop()
        self.receiver_task = loop.create_task(self._receiver_loop())

    # ------------------------------------------------------
    # Pipecat 生命周期：Stop
    # ------------------------------------------------------
    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        self.running = False

        # 发送 finish-task
        if self.ws:
            msg = {
                "header": {"action": "finish-task", "task_id": self.task_id, "streaming": "duplex"},
                "payload": {"input": {}},
            }
            # await self.ws._send(json.dumps(msg))
            await self._safe_send(json.dumps(msg))  # 安全发送 finish-task，不知道是否必要

        # 关闭 websocket
        try:
            if self.receiver_task:
                self.receiver_task.cancel()
            if self.ws:
                await self.ws.close()
        except:
            pass

    # ------------------------------------------------------
    # Pipecat 生命周期：Cancel
    # ------------------------------------------------------
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self.running = False
        try:
            if self.ws:
                await self.ws.close()
        except:
            pass

    # ------------------------------------------------------
    # 安全发送 WebSocket 消息
    # 如果发送失败，会尝试重新连接并发送
    # ------------------------------------------------------
    async def _safe_send(self, data):
        try:
            await self.ws.send(data)
        except Exception:
            print("WebSocket send failed, reconnecting...")
            await self._reconnect()
            await self.ws.send(data)

    # ------------------------------------------------------
    # 将 audio PCM 发送到 FunASR
    # Pipecat 会每帧调用 run_stt(audio_bytes)
    # ------------------------------------------------------
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self.ws:
            await self._safe_send(audio)
        yield None  # Frame 由接收线程产生

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
