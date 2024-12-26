from typing import Any, Literal
from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip
import requests

from config import Config

config = Config()


def query(audio_bytes: bytes) -> dict[str, Any]:
    response = requests.post(
        url="https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
        headers={"Authorization": f"Bearer {config.hf_token}"},
        data=audio_bytes,
    )
    return response.json()


def convert_video_to_audio(video_path: str, output_format: Literal["mp3"] = "mp3") -> str:
    """Convert a video file to an audio file."""
    input_file = Path(video_path)
    output_file = input_file.with_suffix(f".{output_format}")

    if not output_file.exists():
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(output_file.as_posix())

    return output_file.as_posix()


def split_audio_into_chunks(filename: str, max_size_mb: int, is_test: bool) -> list[bytes]:
    """Split an audio file into chunks based on the maximum size (MB)."""
    input_file = Path(filename)
    audio_chunks: list[bytes] = []

    with AudioFileClip(filename) as audio:
        total_duration = audio.duration

        # Calculate maximum duration per chunk based on size constraint
        average_bitrate = audio.reader.infos["audio_bitrate"] * 1024
        max_duration = (max_size_mb * 8 * 1024 * 1024) / average_bitrate

        start_time = 0

        while start_time < total_duration:
            end_time = min(start_time + max_duration, total_duration)

            with audio.subclipped(start_time, end_time) as chunk:
                chunk_path = input_file.with_stem(
                    f"{input_file.stem}_chunk_{len(audio_chunks) + 1}"
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


if __name__ == "__main__":
    video_path = "./data/tmp/sample_41.mp4"

    # Step 1: Convert video to audio
    audio_path = convert_video_to_audio(video_path)

    # Step 2: Split audio into chunks
    audio_chunks = split_audio_into_chunks(audio_path, max_size_mb=5, is_test=False)

    # Step 3: Process each audio chunk
    transcripts = ""
    for audio_chunk in audio_chunks:
        result = query(audio_chunk)
        transcripts += result["text"]
