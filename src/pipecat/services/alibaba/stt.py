"""pipecat.services.alibaba.stt 的 Docstring."""

# alids/stt.py
import asyncio
import os
from typing import AsyncGenerator

from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

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


class AliDSCallback(RecognitionCallback):
    """AliDS 回调，线程安全地调度协程到主线程事件循环."""

    def __init__(self):
        super().__init__()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._user_id: str = "default"
        self.push_frame = None  # 会在服务初始化时注入

    def on_open(self) -> None:
        print("RecognitionCallback open.")

    def on_close(self) -> None:
        print("RecognitionCallback close.")

    def on_complete(self) -> None:
        print("RecognitionCallback completed.")

    def on_error(self, message) -> None:
        print("RecognitionCallback task_id: ", message.request_id)
        print("RecognitionCallback error: ", message.message)

    def on_event(self, result: RecognitionResult) -> None:
        """把识别结果转为 Pipecat Frame 并发送到 pipeline."""
        sentence = result.get_sentence()
        if "text" not in sentence:
            return

        frame = (
            TranscriptionFrame(
                sentence["text"], self._user_id, time_now_iso8601(), language=None, result=sentence
            )
            if RecognitionResult.is_sentence_end(sentence)
            else InterimTranscriptionFrame(
                sentence["text"], self._user_id, time_now_iso8601(), language=None, result=sentence
            )
        )

        # 调度到事件循环
        if self._main_loop:
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self._main_loop)
        else:
            print("Error: no main loop set for AliDSCallback")


class AliDSSTTService(STTService):
    """STT service for Aliyun AliDS using WebSocket API."""

    def __init__(
        self,
        # model: str = "fun-asr-realtime",
        model: str = "paraformer-realtime-v2",
        api_key: str = None,
        sample_rate: int = 16000,
        sample_format: str = "pcm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("ALIDS_API_KEY")
        if not self.api_key:
            raise ValueError("api_key is required for AliDS")

        import dashscope

        dashscope.api_key = self.api_key

        self.callback = AliDSCallback()
        self.recognition = Recognition(
            model=self.model,
            callback=self.callback,
            format=sample_format,
            sample_rate=sample_rate,
            semantic_punctuation_enabled=False,
        )

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._user_id = frame.metadata.get("user_id", "default")

        # 注入回调信息
        self.callback._user_id = self._user_id
        self.callback.push_frame = self.push_frame
        self.callback._main_loop = asyncio.get_running_loop()

        self.recognition.start()

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        self.recognition.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self.recognition.stop()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """把音频送入 AliDS 识别."""
        self.recognition.send_audio_frame(audio)
        yield None  # 由回调处理 Frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """支持 Pipeline 处理."""
        await super().process_frame(frame, direction)


if __name__ == "__main__":
    # 简单测试
    import asyncio

    async def main():
        stt_service = AliDSSTTService(api_key="你的AliDS API Key")
        start_frame = StartFrame(metadata={"user_id": "test_user"})
        await stt_service.start(start_frame)
        # 这里可以调用 run_stt 发送音频
        await asyncio.sleep(5)  # 等待回调打印
        await stt_service.stop(StopFrame())

    asyncio.run(main())
