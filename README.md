<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# GLaDOS Persönlichkeitskern

Ein lokales Sprachassistenten-Projekt, das GLaDOS als Persönlichkeit mit eigenen KI-Modellen realisiert – keine Cloud-Abhängigkeiten, vollständig offline.

**Community:** [Discord beitreten](https://discord.com/invite/ERTDKwpjNB) | **Support:** [Ko-fi](https://ko-fi.com/dnhkng)

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

## Changing the LLM Model

To use other models, use the command:
```ollama pull {modelname}```
and then add it to glados_config.yaml as the model:

    model: "{modelname}"

where __{modelname}__ is a placeholder to be replaced with the model you want to use. You can find [more models here!](https://ollama.com/library)

## Changing the Voice Model

You can use voices from Kokoro too!
Select a voice from the following:
 - ### Female
  - **US**
    - af_alloy
    -  af_aoede
    -  af_jessica
    -  af_kore
    -  af_nicole
    -  af_nova
    -  af_river
    -  af_saraha
    -  af_sky
  - **British**
    - bf_alice
    - bf_emma
    - bf_isabella
    - bf_lily
 - ### Male
  - **US**
    -  am_adam
    -  am_echo
    -  am_eric
    -  am_fenrir
    -  am_liam
    -  am_michael
    -  am_onyx
    -  am_puck
  - **British**
    - bm_daniel
    - bm_fable
    - bm_george
    - bm_lewis

and then add it to glados_config.yaml as the voice, e.g.:

    voice: "af_bella"

## OpenAI-compatible TTS server

To run the OpenAI-compatible TTS server, first install dependencies using the installer script:

   Mac/Linux:

        python scripts/install.py --api

   Windows:

        python scripts\install.py --api

Then run the server with:

    ./scripts/serve

Alternatively, you can run the server in Docker:

    docker compose up -d --build

You can generate voice like this:

    curl -X POST http://localhost:5050/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello world! This is a test.",
        "voice": "glados"
    }' \
    --output speech.mp3

NOTE: The server will not automatically reload on changes when running with Docker. When actively developing, it is recommended to run the server locally using the `serve` script.

The server will be available at [http://localhost:5050](http://localhost:5050)

## More Personalities or LLM's
Make a copy of the file 'configs/glados_config.yaml' and give it a new name, then edit the parameters:

    model:  # the LLM model you want to use, see "Changing the LLM Model"
    personality_preprompt:
    system:  # A description of who the character should be
        - user:  # An example of a question you might ask
        - assistant:  # An example of how the AI should respond
  
To use these new settings, use the command:

    uv run glados start --config configs/assistant_config.yaml

## Common Issues
1. If you find you are getting stuck in loops, as GLaDOS is hearing herself speak, you have two options:
   1. Solve this by upgrading your hardware. You need to you either headphone, so she can't physically hear herself speak, or a conference-style room microphone/speaker. These have hardware sound cancellation, and prevent these loops.
   2. Disable voice interruption. This means neither you nor GLaDOS can interrupt when GLaDOS is speaking. To accomplish this, edit the `glados_config.yaml`, and change `interruptible:` to  `false`.
2. If you get the following error:

    `ImportError: DLL load failed while importing onnxruntime_pybind11_state`
   
   you can fix it by installing the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).


## Testing the submodules
Want to mess around with the AI models? You can test the systems by exploring the 'demo.ipynb'.


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=dnhkng/GlaDOS&type=Date)](https://star-history.com/#dnhkng/GlaDOS&Date)
