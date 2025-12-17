"""pipecat.services.ali.tts 的 Docstring."""

import asyncio
import json
import os
import uuid
from typing import AsyncGenerator

import aiohttp
import websockets
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StopFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

# 尝试导入 TTSAudioRawFrame，如果不存在则回退到 AudioRawFrame
try:
    from pipecat.frames.frames import TTSAudioRawFrame
except ImportError:
    TTSAudioRawFrame = AudioRawFrame

try:
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
except ImportError:
    logger.error("dashscope library not found. Please install it via `pip install dashscope`")

    # 定义伪类防止定义错误
    class SpeechSynthesizer:
        pass

    class AudioFormat:
        pass

    class ResultCallback:
        pass


class AliDSTTSCallback(ResultCallback):
    def __init__(self, service, done_event: asyncio.Event):
        super().__init__()
        self.service = service
        self.done_event = done_event
        # 不立即绑定，service.start 会设置 service._main_loop；回调会尝试优先使用 service._main_loop
        self._loop = None

    def on_open(self):
        logger.debug("AliDSTTSCallback open.")

    def on_close(self):
        logger.debug("AliDSTTSCallback close.")

    def on_complete(self):
        logger.debug("AliDSTTSCallback completed.")
        if self._loop:
            self._loop.call_soon_threadsafe(self.done_event.set)

    def on_error(self, message):
        logger.error(f"AliDSTTSCallback error: {message}")
        if self._loop:
            self._loop.call_soon_threadsafe(self.done_event.set)

    def on_event(self, message):
        pass

    def on_data(self, data: bytes):
        if not data:
            return

        # 选择事件循环：优先使用回调自身已设置的 loop，再使用 service._main_loop，最后尝试当前运行 loop
        loop = self._loop or getattr(self.service, "_main_loop", None)
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

        if loop is None:
            logger.warning("AliDSTTSCallback: no event loop available to push frame")
            return

        # 灵活构造音频帧：兼容 TTSAudioRawFrame/AudioRawFrame 不同签名（positional / named）
        frame = None
        for ctor in (TTSAudioRawFrame, AudioRawFrame):
            try:
                # 先尝试最常见的 positional: ctor(data, sample_rate=..., num_channels=...)
                frame = ctor(data, sample_rate=self.service._sample_rate, num_channels=1)
                break
            except Exception:
                try:
                    # 再尝试常见的 named param: ctor(audio=data, ...)
                    frame = ctor(audio=data, sample_rate=self.service._sample_rate, num_channels=1)
                    break
                except Exception:
                    continue

        if frame is None:
            logger.error("AliDSTTSCallback: failed to construct audio frame from TTS chunk")
            return

        # 防御性补齐属性，避免管道处理时报错或丢弃
        if not hasattr(frame, "id"):
            try:
                frame.id = uuid.uuid4().hex
            except Exception:
                frame.id = None
        if not hasattr(frame, "transport_destination"):
            frame.transport_destination = None
        if not hasattr(frame, "pts"):
            frame.pts = 0

        # 异步安全地调度到 pipecat 事件循环，并捕获异常
        try:
            future = asyncio.run_coroutine_threadsafe(self.service.push_frame(frame), loop)
            # 非必须等待，但捕获可能的异常以便日志可见
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.debug(f"AliDSTTSCallback: push_frame completed with exception: {e}")
        except Exception as e:
            logger.exception(f"AliDSTTSCallback: failed to schedule push_frame: {e}")


class ProductionCosyVoiceTTSService(TTSService):
    def __init__(
        self,
        api_key: str,
        voice_id: str = "longxiaochun",
        model: str = "cosyvoice-v1",
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._sample_rate = sample_rate

        if api_key:
            dashscope.api_key = api_key

        self._main_loop = None

    async def start(self, frame: Frame):
        await super().start(frame)
        self._main_loop = asyncio.get_running_loop()

    async def stop(self, frame: Frame):
        await super().stop(frame)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"TTS Request: {text[:20]}...")

        yield TTSStartedFrame()

        done_event = asyncio.Event()
        callback = AliDSTTSCallback(self, done_event)

        # 将服务的运行循环注入回调，避免回调查找时 race
        if not callback._loop:
            callback._loop = getattr(self, "_main_loop", None)

        # 映射采样率到 DashScope AudioFormat
        audio_format = AudioFormat.PCM_16000HZ_MONO_16BIT
        if self._sample_rate == 22050:
            audio_format = AudioFormat.PCM_22050HZ_MONO_16BIT
        elif self._sample_rate == 24000:
            audio_format = AudioFormat.PCM_24000HZ_MONO_16BIT
        elif self._sample_rate == 8000:
            audio_format = AudioFormat.PCM_8000HZ_MONO_16BIT

        synthesizer = SpeechSynthesizer(
            model=self._model,
            voice=self._voice_id,
            format=audio_format,
            callback=callback,
        )

        try:
            # 使用 streaming_call 发送文本，音频数据将通过 callback.on_data 推送
            synthesizer.streaming_call(text)
            synthesizer.streaming_complete()

            # 等待合成完成
            await done_event.wait()

        except Exception as e:
            logger.error(f"TTS Error: {e}")
            yield ErrorFrame(error=str(e))

        yield TTSStoppedFrame()
