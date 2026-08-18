# GLaDOS Persönlichkeitskern

Ein lokales Sprachassistenten-Projekt, das GLaDOS als Persönlichkeit mit eigenen KI-Modellen realisiert – kann theoretisch offline funktionieren. Das Projekt wurde vom Englischen geforked und es wurde für ein deutschtes Voice Modell angepasst.

## Eigenschaften

- **Spracherkennung (ASR):** Whisper, Parakeet-TDT, oder CTC – lokale ONNX-Modelle
- **Text-zu-Sprache (TTS):** Spezielle GLaDOS-Stimme oder Kokoro-Stimmen – ONNX-basiert
- **LLM-Integration:** Kompatibel mit OpenAI API, Ollama oder anderen lokalen LLM-Servern
- **Sprachaktivitätserkennung:** Silero VAD für automatische Spracherkennung
- **Niedrige Latenz:** Streaming-Architektur, < 600ms Antwortzeit
- **Hochgradig konfigurierbar:** Vollständig anpassbare Persönlichkeit via YAML
- **Plattformübergreifend:** Windows, macOS, Linux – inkl. SBCs (8GB RAM)

## Schnellstart

### Voraussetzungen

- **Python 3.12+**
- **LLM-Server:** Ollama oder OpenAI API (lokale oder Cloud)
- Optional: **NVIDIA CUDA** oder **ROCm** für GPU-Beschleunigung

### Installation & Start

```bash
# Repository klonen
git clone https://github.com/dnhkng/GLaDOS.git
cd GLaDOS

# Installation (lädt Modelle herunter)
python scripts/install.py

# LLM starten (mit Ollama)
ollama pull deepseek-chat

# GLaDOS starten
uv run glados          # Sprachinteraktive Version
uv run glados tui      # Text UI (experimentell)
uv run glados say "Der Kuchen ist real"  # Text aussprechen
```

### API-Server (optional)

```bash
# OpenAI API-kompatible TTS-Schnittstelle
uv run glados api
# POST http://localhost:8000/v1/audio/speech
```

## Architektur

**Audio-Pipeline:**
1. **Continuous Recording** → Zirkulärer Buffer mit Silero VAD
2. **Speech Detection** → Automatische Stummschaltung und Sprachende-Erkennung
3. **Transcription** → ASR-Modell (Whisper/Parakeet) in ONNX
4. **LLM Processing** → Streaming-Response vom LLM-Server
5. **TTS Synthesis** → Satzweise Generierung während der Ausgabe
6. **Audio Playback** → Direktes Streaming an Audio-System

**Komponenten:**
- `core/engine.py` – Haupt-Orchestrator
- `ASR/` – Spracherkennung (Whisper, Parakeet, CTC)
- `TTS/` – Text-zu-Sprache (GLaDOS, Kokoro)
- `audio_io/` – Audio I/O abstraction (sounddevice)
- `api/` – OpenAI-kompatible REST-API
- `core/llm_processor.py` – LLM-Streaming

## Konfiguration

Hauptdatei: **`configs/glados_config.yaml`**

```yaml
Glados:
  llm_model: "deepseek-chat"
  completion_url: "https://api.deepseek.com/chat/completions"  # oder lokal: http://localhost:11434/api/chat
  api_key: "sk-xxx..."
  
  asr_engine: "whisper"        # "whisper", "tdt", "ctc"
  asr_model_path: "small"      # tiny, base, small, medium, large
  
  voice: "glados"              # "glados" oder Kokoro-Stimme
  interruptible: true          # Nutzer kann GLaDOS unterbrechen
  
  personality_preprompt:       # Persönlichkeits-Prompt (GLaDOS-Stil)
    - system: "Du bist GLaDOS..."
```

## Projektstruktur

```
src/glados/
├── core/           # Engine, LLM-Processor
├── ASR/            # Spracherkennung (Whisper, Parakeet, CTC)
├── TTS/            # Text-zu-Sprache
├── audio_io/       # Audio-Systemintegration
├── api/            # REST-API (Optional)
└── utils/          # Helfer & Ressourcen
```

## Roadmap

- [x] Sprachgenerator trainiert & integriert
- [x] Persönlichkeitskern (LLM-Prompt)
- [ ] Speicher-/Kontext-Management
- [ ] Vision (VLM-Integration)
- [ ] Hardware-Animation (Servos/Motoren)
- [ ] 3D-Druck-Designs