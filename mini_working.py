#!/usr/bin/env python3
"""
Minimales Beispiel für deutsches GLaDOS TTS mit Piper
"""

import json
import numpy as np
import onnxruntime as ort
import subprocess
import sys
import wave
from pathlib import Path

def install_espeak():
    """Versuche espeak-ng zu installieren falls nicht vorhanden"""
    try:
        subprocess.run(['espeak-ng', '--version'], capture_output=True, check=True)
        print("espeak-ng bereits installiert")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("espeak-ng nicht gefunden. Installation versuchen...")
        try:
            # Windows: chocolatey
            if sys.platform == "win32":
                subprocess.run(['choco', 'install', 'espeak-ng', '-y'], check=True)
            # Linux: apt
            elif sys.platform == "linux":
                subprocess.run(['sudo', 'apt', 'install', 'espeak-ng', '-y'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Automatische Installation fehlgeschlagen. Bitte espeak-ng manuell installieren:")
            print("Windows: choco install espeak-ng")
            print("Linux: sudo apt install espeak-ng")
            return False

def text_to_phonemes(text: str, lang: str = "de") -> str:
    """Konvertiere Text zu IPA-Phonemen mit espeak-ng"""
    try:
        result = subprocess.run(
            ['espeak-ng', '-q', '--ipa=2', '-v', lang, text],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("espeak-ng nicht gefunden. Bitte installieren Sie espeak-ng")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"espeak-ng Fehler: {e}")

def phonemes_to_ids(phonemes: str, phoneme_id_map: dict) -> list[int]:
    """Konvertiere Phoneme zu IDs"""
    # Spezielle Tokens
    BOS = "^"  # beginning of sentence  
    EOS = "$"  # end of sentence
    PAD = "_"  # padding
    
    if not phonemes or not phonemes.strip():
        print("Warnung: Leere Phoneme - verwende nur BOS/EOS")
        return list(phoneme_id_map[BOS]) + list(phoneme_id_map[EOS])
    
    ids = list(phoneme_id_map[BOS])
    
    unknown_phonemes = []
    for phoneme in phonemes:
        if phoneme in phoneme_id_map:
            ids.extend(phoneme_id_map[phoneme])
            ids.extend(phoneme_id_map[PAD])
        else:
            unknown_phonemes.append(phoneme)
    
    if unknown_phonemes:
        print(f"Warnung: Unbekannte Phoneme ignoriert: {unknown_phonemes}")
    
    ids.extend(phoneme_id_map[EOS])
    print(f"Generierte IDs: {len(ids)} total")
    return ids

def synthesize_audio(phoneme_ids: list[int], model_path: str, config: dict) -> np.ndarray:
    """Synthetisiere Audio aus Phonem-IDs"""
    
    # ONNX Session erstellen
    providers = ort.get_available_providers()
    # Problematische Provider entfernen
    for provider in ["TensorrtExecutionProvider", "CoreMLExecutionProvider"]:
        if provider in providers:
            providers.remove(provider)
    
    ort_session = ort.InferenceSession(model_path, providers=providers)
    
    # Input vorbereiten
    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
    
    # Synthesis-Parameter
    scales = np.array([
        config["inference"]["noise_scale"],
        config["inference"]["length_scale"], 
        config["inference"]["noise_w"]
    ], dtype=np.float32)
    
    # Speaker ID (falls multi-speaker)
    sid = None
    if config["num_speakers"] > 1:
        sid = np.array([0], dtype=np.int64)
    
    # ONNX Inference
    ort_inputs = {
        "input": phoneme_ids_array,
        "input_lengths": phoneme_ids_lengths,
        "scales": scales
    }
    
    if sid is not None:
        ort_inputs["sid"] = sid
        
    audio = ort_session.run(None, ort_inputs)[0].squeeze((0, 1))
    
    return audio

def save_wav(audio: np.ndarray, filename: str, sample_rate: int = 22050):
    """Speichere Audio als WAV-Datei"""
    # Audio zu 16-bit PCM konvertieren
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def main():
    # Pfade zu Modell-Dateien
    model_path = "models/TTS/de_DE-glados-high.onnx"
    config_path = "models/TTS/de_DE-glados-high.onnx.json"
    
    # Prüfe ob Dateien existieren
    if not Path(model_path).exists():
        print(f"Modell nicht gefunden: {model_path}")
        print("Bitte lade das deutsche GLaDOS-Modell herunter")
        return
    
    if not Path(config_path).exists():
        print(f"Konfiguration nicht gefunden: {config_path}")
        return
    
    # espeak-ng prüfen/installieren
    if not install_espeak():
        return
    
    # Konfiguration laden
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Text eingeben
    text = input("Text eingeben (oder Enter für Beispieltext): ").strip()
    if not text:
        text = "Hallo, ich bin GLaDOS. Das deutsche Sprachmodell funktioniert."
    
    print(f"Text: {text}")
    
    try:
        # 1. Text zu Phonemen
        phonemes = text_to_phonemes(text, "de")
        print(f"Phoneme: {phonemes}")
        
        # 2. Phoneme zu IDs
        phoneme_ids = phonemes_to_ids(phonemes, config["phoneme_id_map"])
        print(f"Phonem-IDs: {len(phoneme_ids)} IDs generiert")
        
        # 3. Audio synthetisieren
        print("Synthetisiere Audio...")
        audio = synthesize_audio(phoneme_ids, model_path, config)
        print(f"Audio generiert: {len(audio)} Samples")
        
        # 4. Als WAV speichern
        output_file = "glados_output.wav"
        save_wav(audio, output_file, config["audio"]["sample_rate"])
        print(f"Audio gespeichert: {output_file}")
        
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()