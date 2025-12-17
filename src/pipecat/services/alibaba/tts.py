"""pipecat.services.alibaba.tts 的 Docstring."""

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator

from dashscope.audio.tts_v2 import *

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    StopFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601


def get_timestamp():
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp


class AliDSTTSCallback(ResultCallback):
    """Aliyun AliDSTTS 回调，线程安全地推送 AudioFrame 到 Pipecat."""

    def __init__(self):
        super().__init__()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._user_id: str = "default"
        self.push_frame = None  # 由服务初始化时注入
        self.num_channels = 1
        self.sample_rate = 16000

    def on_open(self):
        print("AliDSTTSCallback open.")

    def on_close(self):
        print("AliDSTTSCallback close.")

    def on_complete(self):
        # print("AliDSTTSCallback completed.")
        print("AliDSTTSCallback 语音合成完成，所有合成结果已被接收：" + get_timestamp())

    def on_error(self, message):
        print("AliDSTTSCallback error:", message)

    def on_event(self, message: str) -> None:
        # print("AliDSTTSCallback event:", message)
        pass

    def on_data(self, data: bytes):
        """把 TTS 音频块转换为 AliDSTTS AudioFrame 并发送."""
        if len(data) == 0:
            return

        frame = TTSAudioRawFrame(
            data,  # bytes PCM
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
        )

        # 调度到事件循环
        if self._main_loop:
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self._main_loop)
        else:
            print("Error: no main loop set for AliDSTTSCallback")


class AliDSTTSService(TTSService):
    """Pipecat TTSService for Aliyun AliDSTTS."""

    def __init__(
        self,
        model: str = "cosyvoice-v2",
        voice: str = "longxiaochun_v2",
        api_key: str = None,
        sample_rate: int = 16000,
        sample_format: str = "pcm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.voice = voice
        self.api_key = api_key or os.environ.get("ALIDS_API_KEY")
        if not self.api_key:
            raise ValueError("api_key is required for AliDSTTS")

        import dashscope

        dashscope.api_key = self.api_key

        self.callback = AliDSTTSCallback()
        self.speechsynthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,  # TODO: 自动根据参数生成
            callback=self.callback,
        )

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._user_id = frame.metadata.get("user_id", "default")
        # TODO，从系统环境变量获取 sample_rate 和 num_channels

        # 注入回调信息
        self.callback._user_id = self._user_id
        self.callback.push_frame = self.push_frame
        self.callback._main_loop = asyncio.get_running_loop()

        # self.speechsynthesizer.start()

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        self.speechsynthesizer.streaming_complete()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self.speechsynthesizer.streaming_cancel()
        # self.speechsynthesizer.streaming_complete()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """发送文本到 AliDSTTS TTS，返回音频流 Frame."""
        # try:
        #     self.speechsynthesizer.streaming_call(text)
        # except Exception as e:
        #     print(f"AliDSTTS error: {e}")
        #     #yield ErrorFrame(f"AliDSTTS error: {e}")
        #     print("AliDSTTS 重新初始化...")
        #     self.speechsynthesizer = SpeechSynthesizer(
        #         model=self.model,
        #         voice=self.voice,
        #         format=AudioFormat.PCM_16000HZ_MONO_16BIT,      # TODO: 自动根据参数生成
        #         callback=self.callback,
        #     )
        #     self.speechsynthesizer.streaming_call(text)
        self.speechsynthesizer.streaming_call(text)

        yield None  # 等待回调推送 Audio 数据

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """支持 Pipeline 处理."""
        await super().process_frame(frame, direction)


if __name__ == "__main__":
    # 简单测试
    import asyncio

    async def main():
        tts_service = AliDSTTSService(api_key="你的AliDSTTS API Key")
        start_frame = StartFrame(metadata={"user_id": "test_user"})
        await tts_service.start(start_frame)
        await tts_service.run_tts("你好，Pipecat！")
        await asyncio.sleep(5)  # 等待回调输出音频
        await tts_service.stop(StopFrame())

    asyncio.run(main())
