# whisper_asr.py
import numpy as np
from pathlib import Path
from typing import Any, Optional
import threading
import queue
import time
import soundfile as sf
from loguru import logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    try:
        import whisper
        FASTER_WHISPER_AVAILABLE = False
    except ImportError:
        raise ImportError("Weder faster-whisper noch openai-whisper installiert. Installiere mit: pip install faster-whisper")


class WhisperASR:
    """
    Whisper-basierte Automatic Speech Recognition für deutsches Audio.
    
    Verwendet faster-whisper für bessere Performance oder fallback auf openai-whisper.
    Optimiert für deutsche Sprache mit schnellen Modellen.
    """
    
    def __init__(
        self,
        model_size: str = "medium",  # medium ist ein guter Kompromiss für Deutsch
        device: str = "auto",        # "cpu", "cuda", oder "auto"
        compute_type: str = "float16",  # "float32", "float16", "int8"
        language: str = "de",        # Sprache auf Deutsch festlegen
        beam_size: int = 1,          # Beam size reduzieren für Geschwindigkeit
        best_of: int = 1,           # Nur ein Kandidat für Geschwindigkeit
        vad_filter: bool = True,     # Voice Activity Detection
        vad_threshold: float = 0.5,  # VAD Schwellwert
        min_silence_duration_ms: int = 500,  # Minimale Stille für Segmentierung
    ):
        """
        Initialisiert das Whisper ASR System.
        
        Args:
            model_size: Whisper Modellgröße ("tiny", "base", "small", "medium", "large")
            device: Compute Device ("cpu", "cuda", "auto")
            compute_type: Precision für faster-whisper ("float32", "float16", "int8")
            language: Sprache für ASR (Standard: "de")
            beam_size: Beam search size (1 = greedy, schneller)
            best_of: Anzahl Kandidaten (1 = schneller)
            vad_filter: Voice Activity Detection aktivieren
            vad_threshold: VAD Schwellwert
            min_silence_duration_ms: Minimale Stille zwischen Segmenten
        """
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Device-Auswahl
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type if self.device == "cuda" else "float32"
        
        # Model laden
        self._load_model()
        
        logger.info(f"WhisperASR initialisiert: {model_size} model auf {self.device}")
        
    def _load_model(self) -> None:
        """Lädt das Whisper-Modell."""
        try:
            if FASTER_WHISPER_AVAILABLE:
                logger.info(f"Lade faster-whisper {self.model_size} model...")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None,  # Standard cache directory
                )
                self.using_faster_whisper = True
            else:
                logger.info(f"Lade openai-whisper {self.model_size} model...")
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device
                )
                self.using_faster_whisper = False
                
            logger.success("Whisper model erfolgreich geladen")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Whisper-Modells: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transkribiert Audio-Daten zu Text.
        
        Args:
            audio_data: Audio als numpy array (float32, mono)
            sample_rate: Sample rate des Audios (Standard: 16000)
            
        Returns:
            str: Transkribierter Text
        """
        try:
            # Audio-Daten vorbereiten
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalisieren falls nötig
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Sample rate check (Whisper erwartet 16kHz)
            if sample_rate != 16000:
                logger.warning(f"Audio hat {sample_rate}Hz, Whisper erwartet 16kHz")
                # Hier könnte Resampling implementiert werden
            
            if self.using_faster_whisper:
                return self._transcribe_faster_whisper(audio_data)
            else:
                return self._transcribe_openai_whisper(audio_data)
                
        except Exception as e:
            logger.error(f"Fehler bei Audio-Transkription: {e}")
            return ""
    
    def _transcribe_faster_whisper(self, audio_data: np.ndarray) -> str:
        """Transkription mit faster-whisper."""
        try:
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=self.beam_size,
                best_of=self.best_of,
                vad_filter=self.vad_filter,
                vad_parameters=dict(
                    threshold=self.vad_threshold,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                ),
                word_timestamps=False,  # Deaktivieren für Geschwindigkeit
            )
            
            # Segmente zu Text zusammenfügen
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                
            transcription = " ".join(text_parts).strip()
            
            if transcription:
                logger.debug(f"Whisper Transkription: '{transcription}'")
                
            return transcription
            
        except Exception as e:
            logger.error(f"Fehler bei faster-whisper Transkription: {e}")
            return ""
    
    def _transcribe_openai_whisper(self, audio_data: np.ndarray) -> str:
        """Transkription mit openai-whisper."""
        try:
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=self.beam_size,
                best_of=self.best_of,
                word_timestamps=False,
            )
            
            transcription = result["text"].strip()
            
            if transcription:
                logger.debug(f"Whisper Transkription: '{transcription}'")
                
            return transcription
            
        except Exception as e:
            logger.error(f"Fehler bei openai-whisper Transkription: {e}")
            return ""
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Verarbeitet einen Audio-Chunk (Kompatibilitätsmethode für bestehenden Code).
        
        Args:
            audio_chunk: Audio chunk als numpy array
            sample_rate: Sample rate
            
        Returns:
            str: Transkribierter Text
        """
        return self.transcribe_audio(audio_chunk, sample_rate)
    
    def is_ready(self) -> bool:
        """
        Prüft ob das ASR-System bereit ist.
        
        Returns:
            bool: True wenn bereit
        """
        return hasattr(self, 'model') and self.model is not None
    
    def cleanup(self) -> None:
        """Cleanup-Methode für Kompatibilität."""
        # faster-whisper und openai-whisper handhaben Cleanup automatisch
        logger.info("WhisperASR cleanup abgeschlossen")
    
    def __del__(self) -> None:
        """Destruktor."""
        try:
            self.cleanup()
        except:
            pass


class WhisperTranscriber:
    """
    Wrapper für WhisperASR, der das TranscriberProtocol implementiert.
    Macht Whisper kompatibel mit dem bestehenden GLaDOS ASR-System.
    """
    
    def __init__(
        self,
        model_path: str | Path = "medium",  # Bei Whisper ist das die Modellgröße
        config_path: str | Path | None = None,  # Für Kompatibilität, wird ignoriert
        model_size: str | None = None,  # Optional: explizite Modellgröße
        device: str = "auto",
        fast_mode: bool = True,
        **kwargs: Any
    ):
        """
        Initialisiert den Whisper Transcriber.
        
        Args:
            model_path: Whisper Modellgröße oder Path (bei Whisper ist das die Größe)
            config_path: Ignoriert für Kompatibilität
            model_size: Explizite Modellgröße (überschreibt model_path)
            device: Computing device ("cpu", "cuda", "auto")
            fast_mode: Geschwindigkeitsoptimierung aktivieren
            **kwargs: Weitere Parameter für WhisperASR
        """
        # Modellgröße bestimmen
        if model_size:
            size = model_size
        elif isinstance(model_path, (str, Path)):
            # Versuche Modellgröße aus Pfad zu extrahieren
            size_str = str(model_path)
            if any(size in size_str for size in ["tiny", "base", "small", "medium", "large"]):
                for size in ["tiny", "base", "small", "medium", "large"]:
                    if size in size_str:
                        size = size
                        break
            else:
                size = "medium"  # Default
        else:
            size = "medium"
            
        logger.info(f"Initialisiere Whisper Transcriber mit Modell: {size}")
        
        # WhisperASR mit optimalen Einstellungen für GLaDOS erstellen
        self.whisper_asr = WhisperASR(
            model_size=size,
            device=device,
            beam_size=1 if fast_mode else 5,
            best_of=1 if fast_mode else 5,
            vad_filter=True,
            compute_type="float16" if device == "cuda" else "float32",
            **kwargs
        )
        
    def transcribe(self, audio_source: np.ndarray) -> str:
        """
        Transkribiert Audio-Daten (implementiert TranscriberProtocol).
        
        Args:
            audio_source: Audio als numpy array
            
        Returns:
            str: Transkribierter Text
        """
        return self.whisper_asr.transcribe_audio(audio_source)
        
    def transcribe_file(self, audio_path: Path) -> str:
        """
        Transkribiert Audio-Datei (implementiert TranscriberProtocol).
        
        Args:
            audio_path: Pfad zur Audio-Datei
            
        Returns:
            str: Transkribierter Text
        """
        try:
            # Lade Audio-Datei
            audio_data, sample_rate = sf.read(audio_path)
            
            # Konvertiere zu mono falls stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            return self.whisper_asr.transcribe_audio(audio_data, sample_rate)
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Audio-Datei {audio_path}: {e}")
            return ""


# Konfigurationsklasse für Whisper
class WhisperConfig:
    """Konfiguration für WhisperASR."""
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "de",
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms


# Factory-Funktion für einfache Erstellung
def create_whisper_asr(
    model_size: str = "medium",
    device: str = "auto",
    fast_mode: bool = True,
) -> WhisperASR:
    """
    Erstellt eine WhisperASR Instanz mit optimalen Einstellungen.
    
    Args:
        model_size: Modellgröße ("tiny", "base", "small", "medium")
        device: Device ("cpu", "cuda", "auto")
        fast_mode: Wenn True, optimiert für Geschwindigkeit
        
    Returns:
        WhisperASR: Konfigurierte ASR-Instanz
    """
    if fast_mode:
        return WhisperASR(
            model_size=model_size,
            device=device,
            beam_size=1,      # Greedy decoding für Geschwindigkeit
            best_of=1,        # Nur ein Kandidat
            vad_filter=True,  # VAD für bessere Segmentierung
            compute_type="float16" if device == "cuda" else "float32",
        )
    else:
        return WhisperASR(
            model_size=model_size,
            device=device,
            beam_size=5,      # Mehr Beam search für Qualität
            best_of=5,        # Mehr Kandidaten
            vad_filter=True,
            compute_type="float32",
        )
