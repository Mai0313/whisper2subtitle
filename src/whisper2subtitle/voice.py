from pathlib import Path

from moviepy import VideoFileClip
import whisper
from pydantic import BaseModel
from rich.console import Console

console = Console()


class VoiceConvertor(BaseModel):
    input_file: Path

    def convert_video_to_audio(self) -> Path:
        """Convert a video file to an audio file."""
        output_file = self.input_file.with_suffix(".mp3")
        if not output_file.exists():
            with VideoFileClip(self.input_file) as video:
                video.audio.write_audiofile(output_file.as_posix())
        self.input_file = output_file
        return self.input_file

    def use_whisper(self) -> whisper.DecodingResult:
        if self.input_file.suffix != ".mp3":
            self.input_file = self.convert_video_to_audio()
        model = whisper.load_model(name="turbo", device="cuda:0")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(self.input_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        console.print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        return result


if __name__ == "__main__":
    vc = VoiceConvertor(input_file="./data/tmp/sample_41.mp4")
    vc.use_whisper()
