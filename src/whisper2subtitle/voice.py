import json
from typing import Any
from pathlib import Path
import warnings
from collections.abc import Iterable

from config import Config
from openai import OpenAI
from moviepy import AudioFileClip, VideoFileClip
import whisper
from pydantic import Field, BaseModel, computed_field
import requests
from rich.console import Console
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import Segment

warnings.filterwarnings("ignore", category=UserWarning)

config = Config()
console = Console()


class VoiceConvertor(BaseModel):
    input_file: Path
    is_test: bool = Field(default=False)

    @computed_field
    @property
    def output_file_srt(self) -> Path:
        return self.input_file.with_suffix(".srt")

    def generate_srt_from_faster_whisper(self, segments: Iterable[Segment]) -> None:
        """將 Faster Whisper 的轉錄結果生成 SRT 文件"""
        srt_content = []
        for i, segment in enumerate(segments):
            start_time = self.format_timestamp(segment.start)
            end_time = self.format_timestamp(segment.end)
            text = segment.text
            srt_content.append(f"{i + 1}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text.strip())
            srt_content.append("")  # 空行分隔

        # 將結果寫入 .srt 文件
        with open(f"{self.output_file_srt}", "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

    def generate_srt_from_whisper(self, result: dict) -> None:
        """將 Whisper 輸出的結果生成 SRT 文件"""
        srt_content = []
        for i, segment in enumerate(result["segments"]):
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            text: str = segment["text"]

            srt_content.append(f"{i + 1}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text.strip())
            srt_content.append("")  # 空行分隔

        # 將結果寫入 .srt 文件
        with open(f"{self.output_file_srt}", "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

    def format_timestamp(self, seconds: float) -> str:
        """格式化時間戳為 SRT 標準格式: HH:MM:SS,mmm"""
        milliseconds = int(seconds * 1000)
        hours = milliseconds // (1000 * 60 * 60)
        milliseconds %= 1000 * 60 * 60
        minutes = milliseconds // (1000 * 60)
        milliseconds %= 1000 * 60
        seconds = milliseconds // 1000
        milliseconds %= 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def convert_video_to_audio(self) -> None:
        """Convert a video file to an audio file."""
        output_file = self.input_file.with_suffix(".mp3")
        if not output_file.exists():
            with VideoFileClip(self.input_file) as video:
                video.audio.write_audiofile(output_file.as_posix())
        self.input_file = output_file

    def split_audio_into_chunks(self, max_size_mb: int) -> list[Path]:
        """Split an audio file into chunks based on the maximum size (MB)."""
        audio_chunks: list[Path] = []

        with AudioFileClip(self.input_file) as audio:
            total_duration = audio.duration

            # Calculate maximum duration per chunk based on size constraint
            average_bitrate = audio.reader.infos["audio_bitrate"] * 1024
            max_duration = (max_size_mb * 8 * 1024 * 1024) / average_bitrate

            start_time = 0

            while start_time < total_duration:
                end_time = min(start_time + max_duration, total_duration)

                with audio.subclipped(start_time, end_time) as chunk:
                    chunk_path = self.input_file.with_stem(
                        f"{self.input_file.stem}_chunk_{len(audio_chunks) + 1}"
                    )
                    chunk_path = chunk_path.with_suffix(".mp3")

                    if isinstance(chunk, AudioFileClip):
                        chunk.write_audiofile(chunk_path.as_posix(), bitrate="50k")

                audio_chunks.append(chunk_path)

                start_time = end_time
                if self.is_test:
                    break

        return audio_chunks

    def use_hugginface_api(self) -> list[dict[str, Any]]:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()

        audio_bytes = self.split_audio_into_chunks(max_size_mb=25)
        transcriptions = []
        for audio in audio_bytes:
            transcription = requests.post(
                url="https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
                headers={"Authorization": f"Bearer {config.hf_token}"},
                data=audio.read_bytes(),
            )
            transcriptions.append(transcription.json())
        return transcriptions

    def use_whisper(self) -> whisper.DecodingResult:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()
        model = whisper.load_model(name="turbo", device="cuda:0")
        result = model.transcribe(self.input_file.as_posix(), word_timestamps=True)

        # For debugging purposes
        with open("./data/result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        self.generate_srt_from_whisper(result)
        return result

    def use_oai_whisper(self) -> list[dict[str, Any]]:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()

        client = OpenAI(api_key=config.openai_api_key)

        audio_bytes = self.split_audio_into_chunks(max_size_mb=25)
        for audio in audio_bytes:
            transcription = client.audio.transcriptions.create(
                file=audio, model="whisper-1", response_format="srt"
            )
            with open(f"{self.output_file_srt}", "w", encoding="utf-8") as f:
                f.write(transcription)
        return transcription

    def use_faster_whisper(self) -> None:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()
        model = WhisperModel("turbo", device="cuda", compute_type="default")
        batched_model = BatchedInferencePipeline(model=model)
        segments, _ = batched_model.transcribe(self.input_file.as_posix(), batch_size=16)
        self.generate_srt_from_faster_whisper(segments)


if __name__ == "__main__":
    vc = VoiceConvertor(input_file="./data/sample_41.mp3", is_test=True)  # type: ignore[arg-type]
    result = vc.use_oai_whisper()
