"""ASR processing components."""

from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray

from .mel_spectrogram import MelSpectrogramCalculator


class TranscriberProtocol(Protocol):
    def __init__(self, model_path: str, *args: str, **kwargs: dict[str, str]) -> None: ...
    def transcribe(self, audio_source: NDArray[Any]) -> str: ...
    def transcribe_file(self, audio_path: Path) -> str: ...


# Factory function
def get_audio_transcriber(
    engine_type: str = "ctc",
    **kwargs: Any,   # <- kwargs wirklich als Any, nicht dict[str, Any]
) -> TranscriberProtocol:
    """
    Factory: erstellt den gewünschten Transcriber und reicht alle kwargs
    (z.B. model_path=..., config_path=...) an den Konstruktor weiter.
    """
    
    if engine_type.lower() == "ctc":
        from .ctc_asr import AudioTranscriber as CTCTranscriber
        return CTCTranscriber(**kwargs)  # <- WICHTIG: kwargs durchreichen
    elif engine_type.lower() == "tdt":
        from .tdt_asr import AudioTranscriber as TDTTranscriber
        return TDTTranscriber(**kwargs)  # <- dito
    elif engine_type.lower() == "whisper":
        from .whisper_asr import WhisperTranscriber
        return WhisperTranscriber(**kwargs)  # <- Whisper hinzufügen
    else:
        raise ValueError(f"Unsupported ASR engine type: {engine_type}")


__all__ = ["MelSpectrogramCalculator", "TranscriberProtocol", "get_audio_transcriber"]
