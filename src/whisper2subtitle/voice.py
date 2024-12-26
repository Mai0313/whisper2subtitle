from typing import Any
from pathlib import Path

from config import Config
from moviepy import AudioFileClip, VideoFileClip
import whisper
from pydantic import BaseModel
import requests
from rich.console import Console

config = Config()
console = Console()


class VoiceConvertor(BaseModel):
    input_file: Path

    def convert_video_to_audio(self) -> None:
        """Convert a video file to an audio file."""
        output_file = self.input_file.with_suffix(".mp3")
        if not output_file.exists():
            with VideoFileClip(self.input_file) as video:
                video.audio.write_audiofile(output_file.as_posix())
        self.input_file = output_file

    def split_audio_into_chunks(self, max_size_mb: int, is_test: bool) -> list[bytes]:
        """Split an audio file into chunks based on the maximum size (MB)."""
        audio_chunks: list[bytes] = []

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

                with open(chunk_path, "rb") as f:
                    audio_chunks.append(f.read())

                start_time = end_time
                if is_test:
                    break

        return audio_chunks

    def use_hugginface_api(self) -> list[dict[str, Any]]:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()

        audio_bytes = self.split_audio_into_chunks(max_size_mb=25, is_test=False)
        responses = []
        for audio in audio_bytes:
            response = requests.post(
                url="https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
                headers={"Authorization": f"Bearer {config.hf_token}"},
                data=audio,
            )
            responses.append(response.json())
        return responses

    def use_whisper(self) -> whisper.DecodingResult:
        if self.input_file.suffix != ".mp3":
            self.convert_video_to_audio()
        model = whisper.load_model(name="turbo", device="cuda:0")
        result = model.transcribe(self.input_file.as_posix())
        return result


if __name__ == "__main__":
    vc = VoiceConvertor(input_file="./data/sample_41.mp3")  # type: ignore[arg-type]
    result = vc.use_whisper()
    console.print(result)
