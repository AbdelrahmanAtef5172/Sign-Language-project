# Real-Time AI Meeting Assistant — Complete Pipeline
## Comprehensive Implementation Guide for CLI-Based LLM Execution

---

> **Document Purpose:** This file is the single source of truth for implementing the complete Real-Time AI Meeting Assistant pipeline. It is designed to be read and followed by a CLI-based LLM (such as Gemini CLI) step by step, without requiring external resources, documentation lookups, or clarifying questions. Every architectural decision, dependency choice, optimization strategy, and implementation order is captured here for all four core components: Vision, Speech-to-Language Recognition (SLR), Large Language Model (LLM), and Text-to-Speech (TTS).

> **Scope Boundary:** This guide covers ALL four pipeline components and their integration. Each component is implemented as an independent module with clean API boundaries. The components are: (1) Vision — Gender + Emotion Detection, (2) SLR — Sign Language Recognition, (3) LLM — Text Refinement and Translation, (4) TTS — Voice Synthesis with Emotional Matching.

> **⚠️ Inference-Only Notice:** This guide describes a **fully inference-based pipeline**. No model training, fine-tuing, or dataset preparation occurs at any point. All models are loaded directly from pretrained weights or API services. If pretrained weight files are not present in their respective `weights/` directories, those components will not function — see weight acquisition sections for each component.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Full Directory and File Structure](#3-full-directory-and-file-structure)
4. [Libraries, Tools, and Dependencies](#4-libraries-tools-and-dependencies)
5. [Environment Setup](#5-environment-setup)
6. [Implementation Pipeline — Phase by Phase](#6-implementation-pipeline--phase-by-phase)
7. [Vision Component Specification](#7-vision-component-specification)
8. [SLR Component Specification](#8-slr-component-specification)
9. [LLM Component Specification](#9-llm-component-specification)
10. [TTS Component Specification](#10-tts-component-specification)
11. [Pipeline Integration and Data Flow](#11-pipeline-integration-and-data-flow)
12. [Session State Management](#12-session-state-management)
13. [WebSocket Server Specification](#13-websocket-server-specification)
14. [Real-Time Optimization Strategy](#14-real-time-optimization-strategy)
15. [Latency Targets and Performance Benchmarks](#15-latency-targets-and-performance-benchmarks)
16. [Testing and Verification Strategy](#16-testing-and-verification-strategy)
17. [Code Quality Standards](#17-code-quality-standards)
18. [Implementation Order and Dependencies](#18-implementation-order-and-dependencies)
19. [Edge Cases and Error Handling Requirements](#19-edge-cases-and-error-handling-requirements)
20. [Completion Checklist](#20-completion-checklist)

---

## 1. Project Overview

### What This Pipeline Does

This is a complete real-time AI pipeline that enables deaf or hard-of-hearing individuals to participate in online meetings through sign language. The system performs four integrated tasks in parallel:

- **Vision Component**: Analyzes webcam video to detect gender (male/female) and emotion (sad/neutral/happy) for TTS voice selection
- **SLR Component**: Recognizes sign language gestures from video frames and converts them to text sequences
- **LLM Component**: Refines recognized sign language text into grammatically correct sentences and optionally translates to target languages
- **TTS Component**: Synthesizes natural speech from refined text using voice profiles matched to detected gender and emotion

### End-to-End User Flow

A deaf participant joins a video meeting with their webcam enabled. The Vision component analyzes their face and determines they are female with a neutral expression. As they sign, the SLR component recognizes individual gestures and assembles them into text fragments. When a complete thought is detected (punctuation boundary or pause), the text is sent to the LLM component for grammatical refinement. The LLM transforms the raw sign language text into a fluent sentence. The TTS component then speaks this sentence aloud using a female voice with neutral emotional tone, allowing hearing participants to understand the contribution without reading captions.

### Why This Pipeline Exists

Sign language is a visual language with its own grammar, word order, and expressiveness. Direct word-for-word transcription often produces choppy, incomplete sentences. This pipeline bridges three gaps: (1) converting visual gestures to text (SLR), (2) transforming sign language grammar into spoken language grammar (LLM), and (3) delivering the message with appropriate vocal characteristics that match the signer's identity and emotional state (TTS with Vision input).

### What This Pipeline Does NOT Do

- It does not perform video conferencing infrastructure (uses existing platforms like Zoom/Meet)
- It does not record or store meeting content
- It does not perform real-time transcription of spoken audio into text
- It does not generate sign language from text (output is audio only)
- It does not train or fine-tune any models at runtime
- It does not require internet connectivity beyond initial API service authentication

### Deployment Model

The pipeline runs as a local FastAPI server with WebSocket endpoints. A browser-based client captures webcam frames and streams them via WebSocket. The server processes frames, maintains session state, and returns audio chunks for playback. All heavy computation (vision inference, SLR recognition, LLM calls, TTS synthesis) happens server-side. The browser only handles camera access and audio playback.

---

## 2. System Architecture

### Architectural Philosophy

The system follows a **modular pipeline architecture with async parallelism**. Each component is a self-contained Python package with no awareness of other components. All integration happens through a central orchestrator that manages WebSocket connections, routes frames to components, and assembles outputs. Dependencies flow downward only — components depend on shared utilities, not on each other.

### Component Layers (Horizontal Separation)

**Vision Component (Layer 1)**
Processes video frames on a background thread every N frames. Updates session state with gender and emotion. Runs independently of SLR component. Never blocks.

**SLR Component (Layer 2)**
Processes video frames on main async queue every frame. Outputs raw text tokens representing recognized signs. Feeds into LLM component when sentence boundaries are detected.

**LLM Component (Layer 3)**
Consumes raw SLR text asynchronously. Refines grammar, adds punctuation, optionally translates. Outputs polished sentences. Feeds into TTS component.

**TTS Component (Layer 4)**
Consumes refined text from LLM and reads gender/emotion from session state. Selects appropriate voice profile. Synthesizes audio chunks. Streams back to client.

**Orchestrator (Layer 0 — Central Hub)**
FastAPI WebSocket server. Manages sessions, routes frames, coordinates component lifecycles. The only layer that knows all components exist.

### Data Flow Through the Pipeline

Raw BGR video frames arrive via WebSocket. The orchestrator clones each frame and dispatches it to two parallel queues: Vision (every 3rd frame on background thread) and SLR (every frame on async queue). Vision updates `SessionState.gender` and `SessionState.emotion`. SLR accumulates recognized signs in a buffer. When a sentence delimiter is detected (period, question mark, pause threshold), SLR pushes the accumulated text to the LLM component. LLM refines and returns a polished sentence. TTS reads the current gender/emotion from SessionState, selects a voice, synthesizes audio, and streams chunks back to the client via WebSocket.

### Parallelism and Concurrency Strategy

The pipeline uses three concurrency mechanisms:

**1. Async/Await for I/O-bound tasks**
WebSocket communication, LLM API calls, TTS synthesis (if using cloud APIs), and SLR inference (if using async-compatible libraries) all use `asyncio`.

**2. Thread Pool for CPU-bound tasks**
Vision component inference runs on `concurrent.futures.ThreadPoolExecutor` via `asyncio.run_in_executor` to avoid blocking the event loop.

**3. Frame Skipping for Real-Time Processing**
Vision processes every 3rd frame. SLR processes every frame but with stride skipping if queue depth exceeds threshold. TTS is triggered only on sentence completion, not per frame.

### Component Independence Guarantees

Each component can be tested in isolation with mock inputs. Vision can run against a video file. SLR can run against a dataset of sign language clips. LLM can process text files. TTS can synthesize from hardcoded sentences. No component directly imports another component. All integration happens through the orchestrator and shared data structures (SessionState, message queues).

---

## 3. Full Directory and File Structure

```
ai_meeting_assistant/
│
├── IMPLEMENTATION_GUIDE.md                  ← this file
├── README.md                                 ← user-facing setup and usage docs
├── setup.py                                  ← pip install -e . for development
├── requirements-cpu.txt                      ← CPU-only dependencies with pinned versions
├── requirements-gpu.txt                      ← GPU-accelerated dependencies with pinned versions
├── .env.example                              ← template for API keys (LLM, TTS cloud services)
├── .gitignore                                ← excludes weights/, logs/, .env
│
├── vision/                                   ← Vision Component Package
│   ├── __init__.py                           ← exports VisionConfig, VisionInferenceEngine
│   ├── config.py                             ← VisionConfig dataclass
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                       ← ViTBackbone (pretrained ViT-Small)
│   │   ├── heads.py                          ← GenderClassificationHead + EmotionClassificationHead
│   │   └── vision_model.py                   ← MultiTaskVisionModel
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── face_detector.py                  ← FaceDetector with MediaPipe
│   │   ├── transforms.py                     ← inference-only transforms
│   │   └── quality_filter.py                 ← QualityFilter for blur/brightness
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py                         ← VisionInferenceEngine (public API)
│   │   ├── temporal_smoother.py              ← TemporalSmoother + PairedSmoother
│   │   └── cache.py                          ← PerceptualHasher + PredictionCache
│   │
│   └── utils/
│       ├── __init__.py
│       ├── onnx_export.py                    ← PyTorch → ONNX export utilities
│       └── visualization.py                  ← debug visualization helpers
│
├── slr/                                      ← Sign Language Recognition Component Package
│   ├── __init__.py                           ← exports SLRConfig, SLRInferenceEngine
│   ├── config.py                             ← SLRConfig dataclass
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pose_extractor.py                 ← MediaPipe Holistic for keypoint extraction
│   │   ├── sequence_encoder.py               ← LSTM/Transformer for temporal modeling
│   │   └── slr_model.py                      ← SLRModel (keypoints → text tokens)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── keypoint_normalizer.py            ← normalize hand/body keypoints to canonical space
│   │   └── sequence_buffer.py                ← sliding window buffer for temporal sequences
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py                         ← SLRInferenceEngine (public API)
│   │   ├── token_decoder.py                  ← beam search decoder for sign tokens
│   │   └── sentence_segmenter.py             ← detects sentence boundaries from pauses
│   │
│   └── utils/
│       ├── __init__.py
│       ├── vocabulary.py                     ← sign token vocabulary and mappings
│       └── visualization.py                  ← draw keypoints on frames for debugging
│
├── llm/                                      ← LLM Refinement Component Package
│   ├── __init__.py                           ← exports LLMConfig, LLMRefinementEngine
│   ├── config.py                             ← LLMConfig dataclass
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── openai_client.py                  ← OpenAI API wrapper (GPT-4, etc.)
│   │   ├── anthropic_client.py               ← Anthropic API wrapper (Claude)
│   │   └── local_client.py                   ← local model wrapper (LLaMA via llama.cpp)
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── refinement_prompts.py             ← system and user prompts for grammar correction
│   │   └── translation_prompts.py            ← prompts for target language translation
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── text_normalizer.py                ← clean and normalize SLR output text
│   │   ├── context_manager.py                ← maintain conversation context across sentences
│   │   └── fallback_handler.py               ← handle API failures gracefully
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py                         ← LLMRefinementEngine (public API)
│   │
│   └── utils/
│       ├── __init__.py
│       └── retry_logic.py                    ← exponential backoff for API calls
│
├── tts/                                      ← Text-to-Speech Component Package
│   ├── __init__.py                           ← exports TTSConfig, TTSEngine
│   ├── config.py                             ← TTSConfig dataclass
│   │
│   ├── voices/
│   │   ├── __init__.py
│   │   ├── voice_profiles.py                 ← defines 6 voice profiles (2 genders × 3 emotions)
│   │   └── voice_selector.py                 ← selects voice based on gender/emotion inputs
│   │
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── coqui_synthesizer.py              ← Coqui TTS local synthesis
│   │   ├── elevenlabs_synthesizer.py         ← ElevenLabs API wrapper
│   │   ├── azure_synthesizer.py              ← Azure TTS API wrapper
│   │   └── synthesizer_interface.py          ← abstract base class for all synthesizers
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── text_preprocessor.py              ← expand abbreviations, normalize numbers
│   │   ├── prosody_adjuster.py               ← adjust speech rate and pitch based on emotion
│   │   └── audio_chunker.py                  ← split long text into streamable chunks
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py                         ← TTSEngine (public API)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio_format.py                   ← WAV/MP3/OGG encoding utilities
│       └── cache.py                          ← cache synthesized audio for repeated phrases
│
├── orchestrator/                             ← Central Orchestration Layer
│   ├── __init__.py                           ← exports OrchestratorApp
│   ├── app.py                                ← FastAPI application setup and lifespan
│   ├── websocket_handler.py                  ← WebSocket route handlers
│   ├── session_manager.py                    ← SessionState management per connection
│   ├── frame_router.py                       ← dispatches frames to Vision and SLR
│   └── audio_streamer.py                     ← streams TTS audio chunks back to client
│
├── shared/                                   ← Shared Utilities Across Components
│   ├── __init__.py
│   ├── session_state.py                      ← SessionState dataclass
│   ├── message_types.py                      ← WebSocket message schemas
│   └── logging_config.py                     ← unified logging setup
│
├── weights/                                  ← Pretrained Model Weights (gitignored)
│   ├── .gitkeep
│   ├── vision/
│   │   ├── gender_head.pt
│   │   └── emotion_head.pt
│   ├── slr/
│   │   └── slr_model.pt
│   └── tts/
│       └── coqui_voices/                     ← Coqui TTS voice model files (if using Coqui)
│
├── logs/                                     ← Runtime Logs (gitignored)
│   └── .gitkeep
│
├── tests/                                    ← Pytest Test Suite
│   ├── __init__.py
│   ├── test_vision/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_preprocessing.py
│   │   ├── test_model.py
│   │   ├── test_smoother.py
│   │   ├── test_cache.py
│   │   └── test_engine.py
│   ├── test_slr/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_keypoint_extraction.py
│   │   ├── test_sequence_buffer.py
│   │   └── test_engine.py
│   ├── test_llm/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_prompts.py
│   │   ├── test_clients.py
│   │   └── test_engine.py
│   ├── test_tts/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_voice_selection.py
│   │   ├── test_synthesizers.py
│   │   └── test_engine.py
│   ├── test_orchestrator/
│   │   ├── __init__.py
│   │   ├── test_session_manager.py
│   │   └── test_websocket_handler.py
│   └── integration/
│       ├── __init__.py
│       └── test_end_to_end.py
│
└── scripts/                                  ← Utility Scripts
    ├── verify_installation.py                ← checks all dependencies and imports
    ├── download_weights.py                   ← downloads pretrained weights from Hugging Face
    ├── benchmark_vision.py                   ← measures Vision component latency
    ├── benchmark_slr.py                      ← measures SLR component latency
    ├── benchmark_llm.py                      ← measures LLM component latency
    ├── benchmark_tts.py                      ← measures TTS component latency
    ├── webcam_demo_vision.py                 ← standalone Vision demo with webcam
    ├── webcam_demo_slr.py                    ← standalone SLR demo with webcam
    ├── test_llm_refinement.py                ← interactive LLM testing script
    ├── test_tts_synthesis.py                 ← interactive TTS testing script
    └── run_server.py                         ← starts the FastAPI server
```

---

## 4. Libraries, Tools, and Dependencies

### Core Framework Dependencies

**Python Version**: 3.10 or 3.11 (required for compatibility with all dependencies)

**Web Server**
- `fastapi==0.104.1` — async web framework for WebSocket server
- `uvicorn[standard]==0.24.0` — ASGI server with WebSocket support
- `websockets==12.0` — WebSocket protocol implementation
- `python-multipart==0.0.6` — form data parsing for file uploads

**Async Runtime**
- `asyncio` (standard library) — event loop and coroutines
- `aiofiles==23.2.1` — async file I/O
- `aiocache==0.12.2` — async caching utilities

### Vision Component Dependencies

**Deep Learning Framework**
- `torch==2.1.0` — PyTorch for model inference
- `torchvision==0.16.0` — pretrained ViT backbone access
- `transformers==4.35.0` — Hugging Face model loading
- `onnxruntime==1.16.1` — ONNX inference engine (CPU)
- `onnxruntime-gpu==1.16.1` — ONNX inference engine (GPU, requirements-gpu.txt only)

**Computer Vision**
- `opencv-python==4.8.1.78` — video frame manipulation
- `mediapipe==0.10.8` — face detection and landmark extraction
- `Pillow==10.1.0` — image preprocessing
- `imagehash==4.3.1` — perceptual hashing for cache

**Utilities**
- `numpy==1.24.3` — array operations
- `scipy==1.11.4` — statistical computations

### SLR Component Dependencies

**Deep Learning Framework**
- `torch==2.1.0` — same as Vision for consistency
- `transformers==4.35.0` — may use pretrained temporal encoders

**Pose Estimation**
- `mediapipe==0.10.8` — Holistic model for hand/body/face keypoints
- `opencv-python==4.8.1.78` — same as Vision

**Sequence Modeling**
- `einops==0.7.0` — tensor operations for sequence handling
- `timm==0.9.12` — may use pretrained video transformers

### LLM Component Dependencies

**API Clients**
- `openai==1.3.5` — OpenAI API client (GPT-4, GPT-3.5)
- `anthropic==0.7.1` — Anthropic API client (Claude 3)
- `tiktoken==0.5.1` — token counting for OpenAI models

**Local Inference (Optional)**
- `llama-cpp-python==0.2.20` — local LLaMA inference via llama.cpp
- `ctransformers==0.2.27` — alternative local model runtime

**Text Processing**
- `langdetect==1.0.9` — language detection for translation routing
- `nltk==3.8.1` — sentence tokenization and text normalization

### TTS Component Dependencies

**Synthesis Engines**
- `TTS==0.20.3` — Coqui TTS for local synthesis (large package, CPU/GPU variants)
- `elevenlabs==0.2.24` — ElevenLabs API client (cloud synthesis)
- `azure-cognitiveservices-speech==1.32.0` — Azure TTS SDK (cloud synthesis)

**Audio Processing**
- `pydub==0.25.1` — audio format conversion and manipulation
- `soundfile==0.12.1` — audio I/O
- `librosa==0.10.1` — audio analysis and effects
- `numpy==1.24.3` — same as Vision

### Shared and Testing Dependencies

**Configuration and Environment**
- `pydantic==2.5.0` — dataclass validation and settings management
- `python-dotenv==1.0.0` — .env file loading for API keys

**Logging and Monitoring**
- `loguru==0.7.2` — structured logging
- `prometheus-client==0.19.0` — metrics export for monitoring

**Testing**
- `pytest==7.4.3` — test framework
- `pytest-asyncio==0.21.1` — async test support
- `pytest-cov==4.1.0` — coverage reporting
- `pytest-mock==3.12.0` — mocking utilities
- `httpx==0.25.1` — async HTTP client for testing WebSocket endpoints

**Development Tools**
- `black==23.11.0` — code formatting
- `isort==5.12.0` — import sorting
- `flake8==6.1.0` — linting
- `mypy==1.7.0` — type checking

### Platform-Specific Considerations

**CPU-Only Setup (requirements-cpu.txt)**
All dependencies listed above with CPU-specific packages where applicable. Suitable for development machines without GPU.

**GPU-Accelerated Setup (requirements-gpu.txt)**
Replace `torch`, `torchvision`, and `onnxruntime` with CUDA-enabled versions. Add `TTS` package built with CUDA support. Requires CUDA 11.8 or 12.1 toolkit installed.

**macOS ARM (M1/M2)**
Use CPU packages but leverage Metal Performance Shaders (MPS) backend for PyTorch where available. TTS synthesis may require Rosetta 2 for some audio codecs.

---

## 5. Environment Setup

### System Requirements

**Minimum Hardware**
- CPU: 4 cores, 2.5GHz or faster (Intel i5/i7 or AMD Ryzen 5/7)
- RAM: 16GB (8GB for OS, 4GB for Vision+SLR models, 2GB for LLM API client, 2GB for TTS)
- Storage: 10GB free (5GB for dependencies, 3GB for model weights, 2GB for logs/cache)
- Webcam: 720p or better, 30fps minimum

**Recommended Hardware**
- CPU: 8 cores, 3.0GHz or faster
- RAM: 32GB
- GPU: NVIDIA RTX 3060 or better with 8GB VRAM (for GPU-accelerated inference)
- Storage: 20GB SSD

**Operating System**
- Linux: Ubuntu 20.04/22.04, Debian 11/12
- macOS: 12.0 (Monterey) or later
- Windows: 10/11 with WSL2 (native Windows support is experimental)

### Python Environment Setup

Create an isolated virtual environment to avoid dependency conflicts.

**Using venv (standard library)**
Navigate to the project root directory. Create a new virtual environment named `venv`. Activate the environment using the appropriate activation script for your operating system. On Linux/macOS this is `source venv/bin/activate`. On Windows this is `venv\Scripts\activate`.

**Using conda (alternative)**
Create a new conda environment named `ai-meeting-assistant` with Python 3.10. Activate the environment with `conda activate ai-meeting-assistant`.

### Dependency Installation

**CPU-Only Installation**
With the virtual environment activated, install dependencies from `requirements-cpu.txt` using pip. Use the `--no-cache-dir` flag to prevent disk space issues on constrained systems. The installation will take 5-10 minutes depending on network speed and machine performance.

**GPU-Accelerated Installation**
First verify CUDA is installed by checking `nvidia-smi` output. CUDA version must match the PyTorch CUDA version specified in `requirements-gpu.txt`. Install dependencies from `requirements-gpu.txt` using pip with `--no-cache-dir` flag. The GPU installation is larger and may take 10-15 minutes.

**Verification**
After installation completes, verify all imports succeed by running `scripts/verify_installation.py`. This script attempts to import every major dependency and reports any failures. All imports must succeed before proceeding.

### API Key Configuration

The LLM and TTS components may require API keys for cloud services. These are stored in a `.env` file at the project root, which is never committed to version control.

**Creating .env File**
Copy `.env.example` to `.env`. Open `.env` in a text editor. Fill in the following keys based on which services you plan to use:

- `OPENAI_API_KEY` — required if using OpenAI GPT models for LLM refinement
- `ANTHROPIC_API_KEY` — required if using Anthropic Claude models for LLM refinement
- `ELEVENLABS_API_KEY` — required if using ElevenLabs for TTS synthesis
- `AZURE_SPEECH_KEY` — required if using Azure TTS
- `AZURE_SPEECH_REGION` — Azure region code (e.g., `eastus`)

Keys you do not use can be left empty or commented out. At minimum, one LLM provider key and one TTS provider key must be configured for the pipeline to function.

**Fallback to Local Models**
If no API keys are provided, the pipeline will attempt to use local models where available. LLM component can fall back to a local LLaMA model if weights are present in `weights/llm/`. TTS component can fall back to Coqui TTS for local synthesis. Local fallback may have reduced quality or speed compared to cloud APIs.

### Model Weights Download

Pretrained model weights are not included in this repository due to size. They must be downloaded separately.

**Automatic Download Script**
Run `scripts/download_weights.py` to automatically download all required weights from Hugging Face and public repositories. This script will download:

- Vision component: gender classification head weights, emotion classification head weights, ViT-Small backbone checkpoint (loaded via transformers library, cached automatically)
- SLR component: sign language recognition model weights for the sequence encoder and decoder
- TTS component: Coqui TTS voice model files if using local synthesis (this is the largest download, approximately 1.5GB)

The script places all weights in the `weights/` directory with the correct subdirectory structure. Download time depends on network speed but typically takes 10-20 minutes.

**Manual Download**
If the automatic script fails, weights can be downloaded manually from the URLs listed in Section 7 (Vision), Section 8 (SLR), and Section 10 (TTS). Place each file in the location specified in the directory structure in Section 3.

### Database and Storage Setup

This pipeline does not use a persistent database. All state is in-memory and tied to WebSocket session lifecycle. Session state is lost when the connection closes or the server restarts.

**Log Directory**
Ensure the `logs/` directory exists and is writable. The `.gitkeep` file should already be present. Logs are rotated daily with a 7-day retention policy. Old logs are automatically deleted.

**Cache Directory**
TTS audio cache is stored in memory by default. For persistent caching across server restarts, set `TTS_CACHE_DIR=/path/to/cache` in `.env`. Cache files will be written to this directory and reloaded on startup.

---

## 6. Implementation Pipeline — Phase by Phase

This section defines the order in which components should be implemented and tested. Each phase is independent and can be verified before moving to the next.

### Phase 1: Foundation and Shared Utilities

**Duration**: 1-2 days

**Deliverables**
- Project directory structure created per Section 3
- All `__init__.py` files in place
- `setup.py` functional for `pip install -e .`
- `requirements-cpu.txt` and `requirements-gpu.txt` populated
- Virtual environment created and dependencies installed
- `scripts/verify_installation.py` passes without errors
- `shared/session_state.py` defined with SessionState dataclass
- `shared/message_types.py` defined with WebSocket message schemas
- `shared/logging_config.py` sets up unified logging with Loguru

**Verification**
Run `pytest tests/ -v` at this point. No component tests will pass yet, but the test discovery should succeed and report all test files found.

### Phase 2: Vision Component

**Duration**: 3-4 days

**Deliverables**
- All Vision component files from Section 7 implemented
- Pretrained weights downloaded and placed in `weights/vision/`
- `VisionConfig` dataclass defined with all constants
- Face detector, quality filter, and transforms functional
- ViT backbone loads from Hugging Face transformers
- Gender and emotion classification heads load from pretrained `.pt` files
- Temporal smoother and cache implemented
- `VisionInferenceEngine` exposes public API
- All Vision component tests in `tests/test_vision/` pass
- `scripts/benchmark_vision.py` reports latency within targets
- `scripts/webcam_demo_vision.py` displays live predictions on webcam feed

**Verification**
Run `pytest tests/test_vision/ -v`. All tests must pass. Run the webcam demo and verify gender and emotion labels update correctly. Run the benchmark script and verify average latency is under 15ms (GPU) or 50ms (CPU).

### Phase 3: SLR Component

**Duration**: 4-5 days

**Deliverables**
- All SLR component files from Section 8 implemented
- Pretrained weights downloaded and placed in `weights/slr/`
- `SLRConfig` dataclass defined with all constants
- MediaPipe Holistic extracts hand/body/face keypoints
- Keypoint normalizer transforms coordinates to canonical space
- Sequence buffer maintains sliding window of keypoints
- Sequence encoder (LSTM or Transformer) processes temporal features
- Token decoder performs beam search to output sign tokens
- Sentence segmenter detects boundaries based on pauses
- `SLRInferenceEngine` exposes public API
- All SLR component tests in `tests/test_slr/` pass
- `scripts/benchmark_slr.py` reports latency within targets
- `scripts/webcam_demo_slr.py` displays recognized signs on webcam feed

**Verification**
Run `pytest tests/test_slr/ -v`. All tests must pass. Run the webcam demo and verify sign tokens appear as you gesture. Run the benchmark script and verify average latency is under 100ms per frame.

### Phase 4: LLM Component

**Duration**: 2-3 days

**Deliverables**
- All LLM component files from Section 9 implemented
- API keys configured in `.env` for at least one provider (OpenAI or Anthropic)
- `LLMConfig` dataclass defined with all constants
- API client wrappers implemented for OpenAI, Anthropic, and local models
- Refinement prompts defined for grammar correction
- Translation prompts defined for target languages (optional)
- Text normalizer cleans SLR output
- Context manager maintains conversation history
- Fallback handler retries failed API calls with exponential backoff
- `LLMRefinementEngine` exposes public API
- All LLM component tests in `tests/test_llm/` pass (may use mock API responses)
- `scripts/test_llm_refinement.py` allows interactive testing with real API

**Verification**
Run `pytest tests/test_llm/ -v`. All tests must pass. Run the interactive testing script and input sample sign language text. Verify the output is grammatically correct and natural-sounding. Test with intentionally broken input to verify error handling.

### Phase 5: TTS Component

**Duration**: 3-4 days

**Deliverables**
- All TTS component files from Section 10 implemented
- Coqui TTS weights downloaded if using local synthesis, or cloud API keys configured
- `TTSConfig` dataclass defined with all constants
- Six voice profiles defined (male/female × sad/neutral/happy)
- Voice selector chooses profile based on gender and emotion inputs
- Synthesizer interface defined as abstract base class
- Coqui, ElevenLabs, and Azure synthesizer implementations complete
- Text preprocessor expands abbreviations and normalizes numbers
- Prosody adjuster modifies speech rate and pitch based on emotion
- Audio chunker splits long text for streaming
- `TTSEngine` exposes public API
- All TTS component tests in `tests/test_tts/` pass
- `scripts/test_tts_synthesis.py` synthesizes sample sentences with all 6 voices
- `scripts/benchmark_tts.py` reports synthesis latency within targets

**Verification**
Run `pytest tests/test_tts/ -v`. All tests must pass. Run the synthesis testing script and listen to audio samples. Verify each voice profile sounds distinct and appropriate for its gender/emotion combination. Run the benchmark script and verify average latency is under 500ms for a 20-word sentence.

### Phase 6: Orchestrator and WebSocket Server

**Duration**: 2-3 days

**Deliverables**
- All orchestrator files from Section 13 implemented
- FastAPI application setup in `orchestrator/app.py`
- WebSocket route handlers defined in `orchestrator/websocket_handler.py`
- Session manager creates and destroys SessionState per connection
- Frame router dispatches frames to Vision and SLR components
- Audio streamer sends TTS chunks back to client
- Lifespan context manager initializes all component engines on startup
- All orchestrator tests in `tests/test_orchestrator/` pass
- `scripts/run_server.py` starts the server successfully

**Verification**
Run `pytest tests/test_orchestrator/ -v`. All tests must pass. Start the server with `scripts/run_server.py`. Connect a WebSocket client (can use a browser console or a tool like `websocat`). Send a mock video frame payload. Verify the server responds without errors and logs show frame routing to components.

### Phase 7: Integration and End-to-End Testing

**Duration**: 2-3 days

**Deliverables**
- Browser-based client implemented (HTML/JavaScript) or use existing test client
- End-to-end integration test in `tests/integration/test_end_to_end.py`
- Test simulates full user flow: send video frames, receive TTS audio chunks
- Performance profiling identifies bottlenecks across all components
- Load testing validates server handles multiple concurrent sessions
- Documentation updated with deployment instructions

**Verification**
Run `pytest tests/integration/ -v`. The end-to-end test must pass. Open the browser client and perform a live demo: sign in front of the camera, verify audio output matches signs, verify voice matches detected gender/emotion. Record latency from sign completion to audio playback start. Target: under 2 seconds total pipeline latency.

---

## 7. Vision Component Specification

The Vision component analyzes webcam video frames to detect the signer's gender and emotional state. This information is used by the TTS component to select an appropriate voice profile.

### Functional Requirements

**Input**
The component accepts BGR video frames as NumPy arrays with shape `(height, width, 3)` and dtype `uint8`. Frames arrive at variable resolutions between 480p and 1080p. The component must handle any frame size without preprocessing errors.

**Output**
The component returns a dictionary containing four fields:
- `gender`: string, either `"male"` or `"female"`
- `emotion`: string, one of `"sad"`, `"neutral"`, or `"happy"`
- `gender_conf`: float between 0.0 and 1.0 representing confidence in gender prediction
- `emotion_conf`: float between 0.0 and 1.0 representing confidence in emotion prediction

**Processing Cadence**
The component processes every third frame only (stride = 3). Frames that are skipped return the last stable prediction from the temporal smoother without any computation.

### Technical Architecture

**Preprocessing Pipeline**
Each incoming frame passes through four stages before reaching the model:

1. Face detection using MediaPipe Face Detection extracts bounding box coordinates for the largest face in frame
2. Face alignment applies affine transformation to center and orient the face using detected landmarks
3. Quality filtering assesses blur and brightness; frames below thresholds are rejected
4. Transform pipeline resizes to 96×96 pixels, normalizes pixel values to [0,1], and converts to PyTorch tensor

**Model Architecture**
The model consists of a shared backbone and two independent classification heads:
- Backbone: ViT-Small (patch size 16, image size 96) pretrained on ImageNet, loaded from `WinKawaks/vit-small-patch16-224` via Hugging Face transformers
- Gender head: single linear layer mapping 384-dim backbone output to 2-dim logits
- Emotion head: single linear layer mapping 384-dim backbone output to 3-dim logits

All backbone parameters have `requires_grad=False`. Only the classification head weights are loaded from pretrained files in `weights/vision/`. No training occurs at runtime.

**Temporal Smoothing**
Raw model predictions can flicker due to frame-to-frame variations in lighting or pose. The temporal smoother maintains a sliding window of the last N predictions for gender and emotion separately. The final output is the majority vote within each window. Gender uses a 30-frame window (slow-changing attribute). Emotion uses a 15-frame window (faster-changing but still stabilized).

**Perceptual Hashing Cache**
If two consecutive frames are nearly identical, recomputing the inference is wasteful. A perceptual hash is computed from the aligned face crop. If the Hamming distance between the new hash and the most recent cached hash is below a threshold, the cached prediction is returned. Cache entries expire after 60 frames to prevent stale predictions.

### Use Case Handling

**Use Case 1: User Joins Meeting**
When the user first opens the webcam, the Vision component receives its first frame. No face is detected initially because the camera is still adjusting exposure. The component returns a result with `data_quality="no_face"`. After a few frames, the face is detected and the smoother begins accumulating predictions. During the first 15-30 frames, confidence values are lower because the smoother has insufficient history. After 30 frames, stable predictions with high confidence are returned.

**Use Case 2: User Switches Emotion Mid-Session**
The user is speaking with a neutral expression, and the Vision component has stabilized on `emotion="neutral"`. The user smiles, and their expression shifts to happy. The next frame that hits the model returns `emotion="happy"` with high confidence, but the temporal smoother still holds 14 "neutral" votes and 1 "happy" vote. The output remains "neutral". Over the next 7-10 frames, "happy" votes accumulate until they form the majority, at which point the output switches to "happy". This prevents brief smirks or fleeting expressions from causing voice profile changes.

**Use Case 3: Lighting Changes**
The user moves closer to a window and lighting on their face becomes harsh, causing one frame to appear overexposed. The quality filter detects the brightness exceeds the threshold and rejects the frame. The inference engine returns the last stable prediction from the smoother without calling the model. Normal lighting resumes on the next frame and processing continues. No voice profile change occurs due to the transient bad frame.

**Use Case 4: No Face Visible**
The user turns away from the camera temporarily. The face detector returns an empty bounding box. The inference engine increments a no-face counter. If the counter exceeds 90 frames (3 seconds at 30fps), the temporal smoother resets its history to avoid stale predictions. The output switches to a default state with low confidence. When the user turns back toward the camera, the smoother re-accumulates predictions from scratch.

### Configuration Constants

All tunable parameters are defined in `vision/config.py` as fields of the `VisionConfig` dataclass:

- `FACE_DETECTION_CONFIDENCE`: minimum confidence threshold for MediaPipe face detection (default 0.5)
- `FACE_MIN_SIZE`: minimum face bounding box size in pixels to avoid detecting distant faces (default 64)
- `QUALITY_BLUR_THRESHOLD`: Laplacian variance threshold below which frames are rejected as too blurry (default 100.0)
- `QUALITY_BRIGHTNESS_RANGE`: acceptable brightness range as (min, max) percentile values (default (20, 235))
- `GENDER_SMOOTHING_WINDOW`: number of frames for gender majority voting (default 30)
- `EMOTION_SMOOTHING_WINDOW`: number of frames for emotion majority voting (default 15)
- `GENDER_CONFIDENCE_THRESHOLD`: minimum softmax probability to accept a gender prediction (default 0.7)
- `EMOTION_CONFIDENCE_THRESHOLD`: minimum softmax probability to accept an emotion prediction (default 0.6)
- `CACHE_HAMMING_THRESHOLD`: maximum Hamming distance to consider two face crops identical (default 5)
- `CACHE_MAX_AGE_FRAMES`: number of frames before cache entries expire (default 60)
- `CACHE_MAX_SIZE`: maximum number of cache entries (LRU eviction, default 100)
- `FRAME_STRIDE`: process every Nth frame only (default 3)
- `NO_FACE_RESET_THRESHOLD`: number of consecutive no-face frames before smoother resets (default 90)

### Integration with Pipeline

The Vision component runs on a background thread pool to avoid blocking the async event loop. When a new frame arrives via WebSocket, the orchestrator calls `vision_engine.process_frame_async(frame)`. This function submits the frame to a `ThreadPoolExecutor` via `asyncio.run_in_executor`. The result is awaited asynchronously and used to update `SessionState.gender` and `SessionState.emotion`. The TTS component reads these fields from SessionState when synthesizing audio. The Vision component never directly communicates with TTS — all coupling is through SessionState.

### Testing Strategy

Tests for the Vision component are located in `tests/test_vision/` and cover:

- Configuration loading and validation
- Face detection on frames with zero, one, or multiple faces
- Face alignment produces correct output shape
- Quality filter correctly rejects blurry and dark frames
- Transform pipeline produces tensors with correct shape and value range
- Model forward pass produces logits with correct dimensions
- Temporal smoother majority voting with various vote distributions
- Cache hits and misses with identical and different face crops
- Inference engine returns correct structure for all edge cases (no face, low quality, stride skip)

All tests must pass before the Vision component is considered complete.

---

## 8. SLR Component Specification

The Sign Language Recognition (SLR) component processes video frames to detect and recognize sign language gestures, converting them into text sequences that represent the signer's intended message.

### Functional Requirements

**Input**
The component accepts BGR video frames as NumPy arrays with shape `(height, width, 3)` and dtype `uint8`. Frames arrive at 30fps. The component processes every frame (no stride skipping) to capture rapid hand movements.

**Output**
The component maintains an internal buffer of recognized sign tokens. When a sentence boundary is detected (pause threshold exceeded or punctuation gesture recognized), the component outputs a string representing the complete sentence. Example output: `"HELLO MY NAME JOHN"`. This raw text is then passed to the LLM component for refinement.

**Processing Cadence**
Unlike the Vision component, SLR must process every frame to avoid missing gestures. However, if the async queue depth exceeds a threshold, frames are dropped to maintain real-time performance. The queue depth threshold is configurable.

### Technical Architecture

**Keypoint Extraction Pipeline**
Each frame passes through MediaPipe Holistic to extract 543 keypoints:

- 468 face landmarks (used for facial grammar markers like questioning expressions)
- 33 body pose landmarks (torso orientation, shoulder position)
- 21 left hand landmarks
- 21 right hand landmarks

Each landmark is represented as (x, y, z, visibility) in normalized coordinates. The keypoint extractor outputs a flat array of shape `(543, 4) = 2172` features per frame.

**Keypoint Normalization**
Raw keypoint coordinates are normalized to a canonical space to handle variations in user distance from camera and body size:

1. Shoulder midpoint becomes the origin (0, 0, 0)
2. Shoulder width defines the scale factor
3. Vertical axis aligns with the spine vector
4. Hand coordinates are relative to wrists

This transformation makes the model invariant to camera position and user body proportions.

**Sequence Buffer**
Sign language is a temporal language — individual frame keypoints are meaningless without context. The sequence buffer maintains a sliding window of the last 32 frames (approximately 1 second at 30fps). Every frame, the oldest frame is dropped and the newest is appended. The full 32-frame sequence is passed to the encoder.

**Sequence Encoder**
The encoder is a temporal model that processes the windowed keypoint sequence:

- Architecture: 2-layer bidirectional LSTM with 512 hidden units per layer, OR
- Architecture: Temporal Transformer with 6 layers, 8 attention heads (alternative implementation)

The encoder outputs a single feature vector of dimension 512 representing the semantic meaning of the gesture sequence. This vector is passed to the token decoder.

**Token Decoder**
The decoder maps the 512-dim feature vector to a probability distribution over the sign vocabulary (approximately 3000 common signs). It uses beam search with beam width 5 to find the most likely sign token. The output is a string token like `"HELLO"` or `"THANK-YOU"`.

**Sentence Segmentation**
Recognized tokens accumulate in a buffer. Sentence boundaries are detected by:

1. Pause detection: if no hand movement detected for 1.5 seconds, end sentence
2. Punctuation gestures: certain signs explicitly indicate sentence end (period, question mark)
3. Buffer overflow: if buffer exceeds 50 tokens, force segmentation

When a boundary is detected, the buffered tokens are concatenated into a single string and emitted to the LLM component. The buffer is then cleared.

### Use Case Handling

**Use Case 1: User Signs Simple Greeting**
User signs `HELLO` then pauses. The keypoint extractor captures hand movements for the `H-E-L-L-O` fingerspelling sequence. The sequence buffer fills with 32 frames showing the progression. The encoder processes the sequence and the decoder recognizes the token `"HELLO"`. The token enters the sentence buffer. The user pauses for 1.5 seconds with hands at rest. The pause detector triggers sentence segmentation. The sentence `"HELLO"` is emitted to the LLM component for refinement.

**Use Case 2: User Signs Complex Sentence**
User signs `MY NAME JOHN I HAPPY MEET YOU` without pauses between tokens. Each token is recognized sequentially and appended to the sentence buffer. After the final sign `YOU`, the user pauses. The pause detector triggers segmentation. The complete sentence `"MY NAME JOHN I HAPPY MEET YOU"` is emitted. The LLM component receives this raw text and refines it to `"My name is John. I'm happy to meet you."`

**Use Case 3: Hand Occlusion**
User's hand briefly passes out of camera frame during a sign. MediaPipe Holistic returns landmarks with low visibility scores for that frame. The keypoint normalizer detects the low visibility and replaces those keypoints with interpolated values from adjacent frames. The sequence encoder receives a slightly noisy but continuous sequence and still recognizes the token correctly.

**Use Case 4: False Gesture Rejection**
User scratches their face — a non-sign movement. The sequence encoder processes the keypoints but the decoder outputs very low probabilities for all tokens (max probability < 0.3). The confidence threshold rejects the prediction and no token is added to the sentence buffer. The gesture is effectively ignored.

**Use Case 5: Rapid Signing**
User signs at a fast pace with minimal pauses between tokens. Tokens accumulate in the sentence buffer faster than the pause detector can trigger. When the buffer reaches 50 tokens, forced segmentation occurs. The sentence is emitted even though the user intended to continue. This is acceptable — long sentences are rare in sign language, and the 50-token limit prevents memory issues. The LLM component can still refine the output correctly.

### Configuration Constants

All tunable parameters are defined in `slr/config.py` as fields of the `SLRConfig` dataclass:

- `HOLISTIC_MIN_DETECTION_CONFIDENCE`: minimum confidence for MediaPipe Holistic pose detection (default 0.5)
- `HOLISTIC_MIN_TRACKING_CONFIDENCE`: minimum confidence for landmark tracking (default 0.5)
- `SEQUENCE_WINDOW_SIZE`: number of frames in sliding window (default 32)
- `ENCODER_TYPE`: `"lstm"` or `"transformer"` (default `"lstm"`)
- `LSTM_HIDDEN_DIM`: hidden dimension for LSTM encoder (default 512)
- `LSTM_NUM_LAYERS`: number of LSTM layers (default 2)
- `TRANSFORMER_NUM_LAYERS`: number of Transformer layers if using Transformer encoder (default 6)
- `TRANSFORMER_NUM_HEADS`: number of attention heads (default 8)
- `VOCABULARY_SIZE`: number of unique sign tokens in vocabulary (default 3000)
- `BEAM_WIDTH`: beam search width for decoding (default 5)
- `TOKEN_CONFIDENCE_THRESHOLD`: minimum decoder probability to accept a token (default 0.3)
- `PAUSE_THRESHOLD_SECONDS`: duration of no movement before sentence segmentation (default 1.5)
- `MAX_SENTENCE_TOKENS`: buffer overflow limit (default 50)
- `QUEUE_DEPTH_LIMIT`: maximum async queue depth before frame dropping (default 10)

### Integration with Pipeline

The SLR component runs on the main async event loop. When a frame arrives via WebSocket, the orchestrator calls `slr_engine.process_frame_async(frame)`. This function is an async coroutine that:

1. Extracts keypoints asynchronously (MediaPipe Holistic may block briefly but typically < 10ms)
2. Updates the sequence buffer
3. If the buffer is full (32 frames), runs encoder and decoder inference
4. If a token is recognized, appends to sentence buffer
5. Checks for sentence boundary conditions
6. If boundary detected, emits the sentence string to the LLM component via a callback

The orchestrator registers a callback function `on_sentence_complete(sentence: str)` that the SLR engine calls when a sentence is ready. This callback forwards the sentence to the LLM component's refinement queue.

### Testing Strategy

Tests for the SLR component are located in `tests/test_slr/` and cover:

- Configuration loading and validation
- Keypoint extraction produces correct shape for frames with visible hands
- Keypoint normalization centers and scales coordinates correctly
- Sequence buffer maintains correct FIFO order and size
- Encoder forward pass produces correct output shape
- Decoder beam search produces valid token indices
- Sentence segmentation triggers on pause threshold
- Sentence segmentation triggers on punctuation gesture
- Confidence thresholding rejects low-probability tokens
- Queue depth limiting drops frames when queue exceeds limit

All tests must pass before the SLR component is considered complete.

---

## 9. LLM Component Specification

The LLM component refines raw sign language text from the SLR component into grammatically correct, natural-sounding sentences. It optionally translates the refined text into a target language if the user has enabled translation.

### Functional Requirements

**Input**
The component accepts raw text strings from the SLR component. Example input: `"MY NAME JOHN I HAPPY MEET YOU"`. Input text may lack punctuation, capitalization, articles, and proper verb conjugation. Input text follows sign language grammar, which differs from spoken language grammar.

**Output**
The component returns refined text as a string. Example output: `"My name is John. I'm happy to meet you."` The output has correct capitalization, punctuation, grammar, and natural phrasing suitable for TTS synthesis.

**Processing Latency**
The component must return results within 2 seconds under normal network conditions when using cloud APIs. Local inference may take up to 5 seconds. Longer latency is acceptable because processing is triggered only at sentence completion, not per frame.

### Technical Architecture

**API Client Layer**
The component supports three backends for LLM inference:

1. OpenAI API (GPT-4 Turbo, GPT-3.5 Turbo) via the `openai` Python client
2. Anthropic API (Claude 3 Sonnet, Claude 3 Opus) via the `anthropic` Python client
3. Local inference (LLaMA 2/3 models) via `llama-cpp-python` or `ctransformers`

Each backend implements a common interface defined in `llm/clients/synthesizer_interface.py` with a single method: `refine(raw_text: str, context: List[str]) -> str`. The interface allows swapping backends without changing calling code.

**Prompt Engineering**
The component uses carefully designed prompts to guide the LLM:

**System Prompt for Refinement**
You are an assistant that converts sign language transcriptions into natural spoken English. The input will be raw sign language text with non-standard grammar, missing articles, and no punctuation. Your task is to:
1. Add appropriate punctuation and capitalization
2. Insert articles (a, an, the) where needed
3. Conjugate verbs correctly
4. Reorder words to match spoken English grammar
5. Preserve the original meaning exactly

Do not add information not present in the input. Do not change the speaker's intent or tone. Output only the refined sentence with no preamble or explanation.

**User Prompt Template**
Input sign language text: `{raw_text}`
Output refined sentence:

**Context Management**
The component maintains a sliding window of the last 5 refined sentences as context. This context is prepended to each refinement request to help the LLM maintain consistency in pronoun usage, verb tense, and topic continuity across multiple sentences. Example:

Previous sentences:
1. "My name is John."
2. "I work at the hospital."
3. "I am a doctor."

Current input: `I HELP PATIENT EVERY DAY`

The LLM uses the context to understand "I" refers to John the doctor, and refines to: `"I help patients every day."`

**Fallback and Retry Logic**
API calls can fail due to network errors, rate limits, or service outages. The component implements exponential backoff retry with jitter:

1. First failure: retry after 1 second
2. Second failure: retry after 2 seconds
3. Third failure: retry after 4 seconds
4. Max retries: 3 attempts

If all retries fail, the component falls back to a simple rule-based text normalizer that adds basic punctuation and capitalization but does not fix grammar. The output quality is lower but the pipeline continues functioning.

**Translation (Optional Feature)**
If translation is enabled in the configuration, the component performs a second LLM call after refinement:

**System Prompt for Translation**
You are a translation assistant. Translate the following English sentence to {target_language}. Preserve the tone and meaning. Output only the translation with no preamble or explanation.

**User Prompt Template**
English: `{refined_text}`
{target_language}:

Translation is always performed on the refined English text, never on the raw sign language text. This ensures high translation quality.

### Use Case Handling

**Use Case 1: Simple Grammar Correction**
Input from SLR: `"HELLO MY NAME SARAH"`
Context: empty (first sentence)
LLM refines to: `"Hello, my name is Sarah."`
Output sent to TTS component.

**Use Case 2: Complex Sentence with Missing Articles**
Input from SLR: `"I GO STORE BUY MILK YESTERDAY"`
Context: previous sentences about Sarah's daily routine
LLM refines to: `"I went to the store to buy milk yesterday."`
Note the LLM added `"the"` and `"to"`, conjugated `"go"` to `"went"`, and inferred past tense from `"yesterday"`.

**Use Case 3: Multi-Sentence Context Consistency**
Input 1: `"MY DOG NAME MAX"`
LLM refines to: `"My dog's name is Max."`

Input 2: `"HE BROWN WHITE"`
Context includes previous sentence about Max
LLM refines to: `"He is brown and white."`
Note the LLM used context to understand `"HE"` refers to Max the dog, not a person.

**Use Case 4: Translation to Spanish**
Input: `"I NEED HELP"`
LLM refines to: `"I need help."`
Translation enabled with target language `es`
LLM translates to: `"Necesito ayuda."`
Output: `"Necesito ayuda."` sent to TTS component.

**Use Case 5: API Failure and Fallback**
Input: `"TODAY GOOD DAY"`
First API call fails (network timeout)
Retry after 1 second fails (service unavailable)
Retry after 2 seconds fails (rate limit)
All retries exhausted
Fallback normalizer activates
Output: `"Today good day."` (basic capitalization added, grammar not fixed)
Output sent to TTS component with degraded quality but no pipeline failure.

### Configuration Constants

All tunable parameters are defined in `llm/config.py` as fields of the `LLMConfig` dataclass:

- `PROVIDER`: one of `"openai"`, `"anthropic"`, `"local"` (default `"openai"`)
- `OPENAI_MODEL`: model identifier for OpenAI (default `"gpt-3.5-turbo"`)
- `ANTHROPIC_MODEL`: model identifier for Anthropic (default `"claude-3-sonnet-20240229"`)
- `LOCAL_MODEL_PATH`: path to local LLaMA model file (default `"weights/llm/llama-2-7b.gguf"`)
- `MAX_TOKENS`: maximum tokens in LLM response (default 100)
- `TEMPERATURE`: sampling temperature for generation (default 0.3, low for consistency)
- `CONTEXT_WINDOW_SIZE`: number of previous sentences to include as context (default 5)
- `RETRY_MAX_ATTEMPTS`: maximum number of retry attempts on API failure (default 3)
- `RETRY_BASE_DELAY`: base delay in seconds for exponential backoff (default 1.0)
- `ENABLE_TRANSLATION`: boolean, whether to perform translation after refinement (default False)
- `TARGET_LANGUAGE`: ISO 639-1 language code for translation (default `"es"` for Spanish)
- `FALLBACK_MODE`: one of `"normalizer"`, `"passthrough"` (default `"normalizer"`)

### Integration with Pipeline

The LLM component runs asynchronously and is invoked by the orchestrator when the SLR component emits a complete sentence. The call flow is:

1. SLR component detects sentence boundary
2. SLR calls registered callback `on_sentence_complete("RAW TEXT")`
3. Orchestrator receives callback and calls `llm_engine.refine_async("RAW TEXT")`
4. LLM component makes API call (async, does not block)
5. LLM returns refined text
6. Orchestrator calls `tts_engine.synthesize_async(refined_text)`

The LLM component never directly interacts with SLR or TTS — all coordination is handled by the orchestrator. The LLM maintains its own context buffer which is independent of SessionState.

### Testing Strategy

Tests for the LLM component are located in `tests/test_llm/` and cover:

- Configuration loading and validation
- API client initialization for all three providers
- Prompt template formatting with various inputs
- Context window maintains correct size and order
- Retry logic executes correct number of attempts with correct delays
- Fallback normalizer activates after max retries
- Translation produces output in correct target language (may use mock API responses)
- Mock API tests for all clients to avoid real API costs during CI

All tests must pass before the LLM component is considered complete.

---

## 10. TTS Component Specification

The Text-to-Speech (TTS) component synthesizes natural-sounding speech from refined text, using voice profiles matched to the signer's detected gender and emotional state. The audio is streamed back to the client for playback.

### Functional Requirements

**Input**
The component accepts refined text strings from the LLM component and reads gender/emotion values from SessionState. Example input: `"My name is John. I'm happy to meet you."` with `SessionState.gender="male"` and `SessionState.emotion="happy"`.

**Output**
The component returns audio chunks as bytes in a streamable format (WAV, MP3, or OGG). Chunks are emitted progressively to reduce time-to-first-audio. Total audio duration matches the text length at natural speaking rate (approximately 150 words per minute).

**Voice Profile Selection**
The component maintains 6 voice profiles corresponding to all combinations of gender and emotion:
- Male + Sad
- Male + Neutral
- Male + Happy
- Female + Sad
- Female + Neutral
- Female + Happy

Each profile is either a distinct voice model (for local synthesis) or a specific voice ID and prosody configuration (for cloud APIs).

**Processing Latency**
The component must emit the first audio chunk within 500ms of receiving text. Total synthesis time for a 20-word sentence should not exceed 2 seconds. Cloud APIs typically achieve 300-800ms latency. Local synthesis may take 1-2 seconds.

### Technical Architecture

**Voice Profile Management**
Voice profiles are defined in `tts/voices/voice_profiles.py` as a dictionary mapping `(gender, emotion)` tuples to profile configurations. Each profile contains:

- `voice_id`: identifier for the voice (model name for local, voice ID string for cloud)
- `pitch_shift`: semitone adjustment from base pitch (e.g., -2 for sad, 0 for neutral, +2 for happy)
- `speed_factor`: speaking rate multiplier (e.g., 0.9 for sad, 1.0 for neutral, 1.1 for happy)
- `intensity`: emotional intensity parameter (0.0 to 1.0, used by some APIs)

**Voice Selector**
The voice selector (`tts/voices/voice_selector.py`) reads `SessionState.gender` and `SessionState.emotion` and looks up the corresponding profile in the voice profiles dictionary. If the gender or emotion value is invalid or missing, it defaults to male + neutral.

**Synthesizer Interface**
The component supports three synthesis backends:

1. Coqui TTS (local, open-source) — high quality, GPU-accelerated, ~1-2 second latency
2. ElevenLabs API (cloud) — very high quality, ~300-500ms latency, requires API key
3. Azure Cognitive Services (cloud) — high quality, ~400-800ms latency, requires API key

Each backend implements a common interface defined in `tts/synthesis/synthesizer_interface.py`:

```
class SynthesizerInterface:
    def synthesize(text: str, voice_config: VoiceProfile) -> AsyncGenerator[bytes]:
        """
        Synthesize text to audio chunks.
        Yields audio bytes progressively.
        """
```

The interface returns an async generator that yields audio chunks as they are produced, enabling streaming to the client.

**Text Preprocessing**
Before synthesis, text passes through preprocessing steps:

1. Abbreviation expansion: `"Dr. Smith"` → `"Doctor Smith"`
2. Number normalization: `"123"` → `"one hundred twenty-three"`
3. Symbol expansion: `"$50"` → `"fifty dollars"`
4. Sentence splitting: long paragraphs are split into individual sentences for chunked synthesis

**Prosody Adjustment**
After selecting a voice profile, the prosody adjuster modifies synthesis parameters based on emotion:

- Sad: reduce speed to 0.9×, lower pitch by 2 semitones, reduce volume slightly
- Neutral: no adjustments
- Happy: increase speed to 1.1×, raise pitch by 2 semitones, increase volume slightly

These adjustments enhance the emotional match between the voice and the detected emotion.

**Audio Format and Encoding**
The component outputs audio in WAV format (16-bit PCM, 22.05kHz sample rate) by default. For lower bandwidth, MP3 or OGG encoding can be enabled. Encoding is performed using `pydub` after synthesis.

**Caching**
Frequently repeated phrases (e.g., greetings, common responses) are cached after synthesis. If the same text is requested again, the cached audio is returned immediately without re-synthesizing. Cache uses LRU eviction with a maximum of 100 entries. Cache is in-memory by default but can be persisted to disk if configured.

### Use Case Handling

**Use Case 1: First Synthesis Request**
User signs and the LLM refines text to `"Hello, how are you?"`. SessionState shows `gender="female"` and `emotion="neutral"`. TTS component:

1. Voice selector chooses Female + Neutral profile
2. Text preprocessor expands abbreviations (none in this case)
3. Prosody adjuster applies neutral settings (no changes)
4. Synthesizer (e.g., ElevenLabs) is called with the selected voice
5. First audio chunk arrives after ~400ms
6. Additional chunks stream progressively
7. Chunks are sent to client via WebSocket
8. Client plays audio immediately upon receiving first chunk
9. Synthesized audio is cached for this text

**Use Case 2: Emotion Change During Session**
User's emotion changes from neutral to happy mid-conversation. Next synthesis request uses the same text `"Thank you"`. SessionState now shows `emotion="happy"`. TTS component:

1. Voice selector chooses Female + Happy profile (different from previous)
2. Text preprocessor runs (no changes for this simple text)
3. Prosody adjuster increases speed to 1.1× and raises pitch by +2 semitones
4. Synthesizer generates audio with happier characteristics
5. Audio is cached separately under the new voice profile (same text, different voice)

**Use Case 3: Long Sentence Chunking**
LLM produces a long refined sentence: `"I went to the store yesterday to buy groceries, and I also stopped by the pharmacy to pick up my prescription."` TTS component:

1. Text preprocessor splits into two sentences based on punctuation: `"I went to the store yesterday to buy groceries,"` and `"and I also stopped by the pharmacy to pick up my prescription."`
2. Each sentence is synthesized separately
3. Audio chunks stream for first sentence immediately
4. Once first sentence audio completes, second sentence synthesis begins
5. Client receives chunks for both sentences sequentially and plays them as one continuous audio stream

**Use Case 4: API Failure and Fallback**
ElevenLabs API call times out due to network issues. TTS component:

1. Detects timeout exception
2. Logs warning
3. Falls back to Coqui TTS local synthesizer
4. Local synthesis completes successfully (slower, but functional)
5. Audio chunks stream to client
6. No user-visible error, just slightly longer latency

**Use Case 5: Cache Hit**
User signs the same greeting they signed 2 minutes ago: `"Hello"`. LLM refines to `"Hello."`. TTS component:

1. Voice selector chooses current voice profile (Female + Neutral)
2. Cache lookup with key `(text="Hello.", voice_profile=female_neutral)` returns cached audio bytes
3. Cached audio is streamed immediately with near-zero latency
4. No synthesis API call is made

### Configuration Constants

All tunable parameters are defined in `tts/config.py` as fields of the `TTSConfig` dataclass:

- `PROVIDER`: one of `"coqui"`, `"elevenlabs"`, `"azure"` (default `"elevenlabs"`)
- `COQUI_MODEL_PATH`: path to Coqui TTS model checkpoint (default `"weights/tts/coqui_voices/en/ljspeech/best_model.pth"`)
- `ELEVENLABS_VOICE_IDS`: dictionary mapping (gender, emotion) to ElevenLabs voice IDs
- `AZURE_VOICE_NAMES`: dictionary mapping (gender, emotion) to Azure voice names
- `AUDIO_FORMAT`: one of `"wav"`, `"mp3"`, `"ogg"` (default `"wav"`)
- `SAMPLE_RATE`: audio sample rate in Hz (default 22050)
- `PITCH_SHIFT_SEMITONES`: dictionary mapping emotion to pitch shift values (default `{"sad": -2, "neutral": 0, "happy": 2}`)
- `SPEED_FACTORS`: dictionary mapping emotion to speed multipliers (default `{"sad": 0.9, "neutral": 1.0, "happy": 1.1}`)
- `ENABLE_CACHE`: boolean, whether to cache synthesized audio (default True)
- `CACHE_MAX_SIZE`: maximum number of cache entries (default 100)
- `CACHE_PERSIST_DIR`: optional directory path for persistent cache (default None, in-memory only)

### Integration with Pipeline

The TTS component is invoked by the orchestrator after the LLM component returns refined text. The call flow is:

1. LLM component returns refined text to orchestrator
2. Orchestrator reads current `SessionState.gender` and `SessionState.emotion`
3. Orchestrator calls `tts_engine.synthesize_async(refined_text)`
4. TTS engine selects voice profile based on SessionState
5. TTS synthesizer yields audio chunks asynchronously
6. Orchestrator forwards each chunk to the WebSocket audio streamer
7. Audio streamer sends chunks to client via WebSocket as binary messages
8. Client receives chunks and plays audio progressively

The TTS component reads from SessionState but does not write to it. It has no direct communication with Vision, SLR, or LLM components.

### Testing Strategy

Tests for the TTS component are located in `tests/test_tts/` and cover:

- Configuration loading and validation
- Voice profile definitions contain all 6 required profiles
- Voice selector correctly maps (gender, emotion) to profiles
- Voice selector uses default profile for invalid inputs
- Text preprocessor expands abbreviations correctly
- Text preprocessor normalizes numbers correctly
- Prosody adjuster applies correct parameters for each emotion
- Synthesizer interface returns async generator of bytes
- Cache correctly stores and retrieves audio
- Cache eviction works correctly when max size exceeded
- Mock synthesis tests for all three providers to avoid real API costs during CI

All tests must pass before the TTS component is considered complete.

---

## 11. Pipeline Integration and Data Flow

This section describes how all four components connect and exchange data to form a complete real-time pipeline.

### High-Level Data Flow Diagram

The pipeline has two parallel input streams (Vision and SLR) and one sequential output stream (LLM → TTS):

```
Video Frame Input (from WebSocket client)
         |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
   [Frame Clone 1]     [Frame Clone 2]   [Original Frame]
         |                  |                  |
         v                  v                  v
  Vision Component     SLR Component     (Discarded)
   (every 3rd frame,   (every frame,
    background thread) async queue)
         |                  |
         v                  v
  Update SessionState    Accumulate Tokens
  (gender, emotion)      in Sentence Buffer
         |                  |
         |                  v
         |            Detect Sentence
         |             Boundary
         |                  |
         |                  v
         |            Emit Sentence String
         |                  |
         |                  v
         |            LLM Component
         |          (refine grammar)
         |                  |
         |                  v
         |            Refined Text
         |                  |
         +------------------+
                            |
                            v
                   Read SessionState
                  (gender, emotion)
                            |
                            v
                      TTS Component
                   (synthesize audio)
                            |
                            v
                      Audio Chunks
                            |
                            v
                   WebSocket Streamer
                            |
                            v
                   Client Audio Playback
```

### Component Triggering Conditions

**Vision Component Triggered:**
- New frame arrives via WebSocket AND frame counter % 3 == 0
- Processing happens on background thread pool
- Result updates SessionState asynchronously

**SLR Component Triggered:**
- New frame arrives via WebSocket (every frame)
- Processing happens on async event loop
- Token accumulates in sentence buffer

**LLM Component Triggered:**
- SLR component detects sentence boundary (pause or punctuation gesture)
- SLR emits sentence string via callback
- Orchestrator receives callback and invokes LLM async

**TTS Component Triggered:**
- LLM component returns refined text
- Orchestrator receives refined text
- Orchestrator reads SessionState for current gender/emotion
- Orchestrator invokes TTS async

### SessionState Structure

The SessionState dataclass holds per-session metadata and component outputs. It is created when a WebSocket connection is established and destroyed when the connection closes.

**SessionState Fields:**
- `session_id`: unique identifier for this WebSocket connection (UUID)
- `gender`: string, current gender label from Vision component (`"male"` or `"female"`)
- `emotion`: string, current emotion label from Vision component (`"sad"`, `"neutral"`, or `"happy"`)
- `gender_conf`: float, confidence score for gender prediction (0.0 to 1.0)
- `emotion_conf`: float, confidence score for emotion prediction (0.0 to 1.0)
- `slr_token_buffer`: list of strings, accumulated sign tokens awaiting sentence boundary
- `llm_context`: list of strings, last N refined sentences for LLM context
- `frame_count`: int, total frames processed for this session
- `last_audio_timestamp`: float, timestamp of last TTS audio chunk sent (for latency tracking)

**Thread Safety:**
SessionState is accessed from multiple async tasks and the Vision background thread. Python's GIL prevents torn reads/writes of simple types (str, float, int). For list operations (appending to `slr_token_buffer` or `llm_context`), the orchestrator uses asyncio locks to prevent race conditions.

### Message Schemas for WebSocket Communication

**Client → Server Messages:**

1. Video Frame Message (binary)
   - Type: binary message
   - Format: raw BGR frame bytes, dimensions encoded in first 8 bytes as two int32 values (height, width)
   - Frequency: 30 times per second

2. Control Message (JSON)
   - Type: text message
   - Schema: `{"type": "control", "action": "pause" | "resume" | "disconnect"}`
   - Frequency: infrequent, user-initiated

**Server → Client Messages:**

1. Audio Chunk Message (binary)
   - Type: binary message
   - Format: raw audio bytes in WAV format
   - Frequency: variable, depends on TTS synthesis speed and text length

2. Status Message (JSON)
   - Type: text message
   - Schema: `{"type": "status", "gender": str, "emotion": str, "recognized_text": str}`
   - Frequency: sent after each sentence is recognized and refined
   - Purpose: display recognized text and current voice settings to user

3. Error Message (JSON)
   - Type: text message
   - Schema: `{"type": "error", "component": str, "message": str}`
   - Frequency: only on component failures
   - Purpose: notify user of degraded functionality

### Async Coordination Patterns

**Vision Component (Background Thread)**
The orchestrator submits frames to a ThreadPoolExecutor:

```
executor = ThreadPoolExecutor(max_workers=1)
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(executor, vision_engine.process_frame, frame)
session_state.gender = result["gender"]
session_state.emotion = result["emotion"]
```

**SLR Component (Async Queue)**
Frames are pushed to an asyncio.Queue and processed by a dedicated consumer coroutine:

```
async def slr_consumer():
    while True:
        frame = await slr_queue.get()
        tokens = await slr_engine.process_frame_async(frame)
        if boundary_detected:
            sentence = "".join(tokens)
            await llm_queue.put(sentence)
```

**LLM Component (Async Processing)**
Sentences from SLR are pushed to another queue and refined asynchronously:

```
async def llm_consumer():
    while True:
        raw_sentence = await llm_queue.get()
        refined = await llm_engine.refine_async(raw_sentence)
        await tts_queue.put(refined)
```

**TTS Component (Async Streaming)**
Refined text is synthesized and audio chunks are streamed back:

```
async def tts_consumer():
    while True:
        refined_text = await tts_queue.get()
        async for audio_chunk in tts_engine.synthesize_async(refined_text):
            await websocket.send_bytes(audio_chunk)
```

### Latency Breakdown and Optimization

**Total Pipeline Latency Target:** Under 2.5 seconds from sign completion to audio playback start

**Component Latency Contributions:**
- Vision: 15ms average (GPU) or 50ms (CPU) — runs in parallel, does not contribute to critical path
- SLR: 50-100ms per frame, sentence detection adds 1.5 seconds pause threshold — critical path
- LLM: 500-1500ms API call latency — critical path
- TTS: 300-800ms for first chunk (cloud) or 1-2 seconds (local) — critical path

**Critical Path:** SLR pause detection (1.5s) + LLM refinement (0.5-1.5s) + TTS first chunk (0.3-0.8s) = 2.3-3.8 seconds

**Optimization Strategies:**
1. Reduce SLR pause threshold to 1.0 second (tradeoff: more false sentence boundaries)
2. Use faster LLM model (GPT-3.5 instead of GPT-4) for lower latency at slight quality cost
3. Use cloud TTS (ElevenLabs) instead of local Coqui for faster synthesis
4. Prefetch LLM context before sentence completion to reduce processing time
5. Stream TTS audio chunks to client as they are generated, not waiting for complete audio

---

## 12. Session State Management

The Session State Management system maintains per-connection state and coordinates component interactions throughout a user's session.

### SessionState Lifecycle

**Creation**
When a client connects to the WebSocket endpoint at `/ws`, the orchestrator creates a new SessionState instance. The `session_id` is generated as a UUID. All state fields are initialized to defaults:

- `gender`: `"male"` (default until Vision component detects otherwise)
- `emotion`: `"neutral"` (default)
- `gender_conf`: 0.0
- `emotion_conf`: 0.0
- `slr_token_buffer`: empty list
- `llm_context`: empty list
- `frame_count`: 0

**Updates During Session**
As frames are processed, various components update SessionState fields:

- Vision component updates `gender`, `emotion`, `gender_conf`, `emotion_conf` every 3rd frame
- SLR component appends to `slr_token_buffer` when tokens are recognized
- SLR component clears `slr_token_buffer` after sentence boundary is detected
- LLM component appends refined sentences to `llm_context` (maintains last 5 entries)
- Orchestrator increments `frame_count` for every frame received

**Destruction**
When the WebSocket connection closes (client disconnects or network error), the orchestrator:

1. Stops all component consumer coroutines for this session
2. Clears all queues (slr_queue, llm_queue, tts_queue) to prevent processing orphaned data
3. Deletes the SessionState instance from the session manager
4. Logs session statistics (total frames processed, total sentences refined, average latencies)

### Concurrency and Thread Safety

SessionState is accessed from multiple async tasks running on the event loop and from the Vision background thread. Race conditions are possible on list mutations.

**Locking Strategy**
The orchestrator wraps all list append/clear operations with asyncio.Lock:

```
async with session_state.slr_lock:
    session_state.slr_token_buffer.append(new_token)

async with session_state.llm_lock:
    session_state.llm_context.append(refined_sentence)
    if len(session_state.llm_context) > 5:
        session_state.llm_context.pop(0)
```

Simple field assignments (gender, emotion, floats) do not require locking due to Python GIL.

### Session Manager Implementation

The session manager (`orchestrator/session_manager.py`) maintains a dictionary mapping `session_id` to SessionState:

```
class SessionManager:
    def __init__(self):
        self.sessions = {}  # session_id -> SessionState
        self.lock = asyncio.Lock()
    
    async def create_session(self) -> str:
        async with self.lock:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = SessionState(session_id=session_id)
            return session_id
    
    async def get_session(self, session_id: str) -> SessionState:
        return self.sessions.get(session_id)
    
    async def destroy_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
```

The session manager is instantiated once in `app.py` and stored in `app.state` for access from WebSocket handlers.

---

## 13. WebSocket Server Specification

The WebSocket server provides the network interface between browser clients and the pipeline components. It handles connection management, frame routing, and audio streaming.

### Server Technology Stack

**Framework:** FastAPI with native WebSocket support
**ASGI Server:** Uvicorn with `--ws websockets` for WebSocket protocol handling
**Concurrency:** Single-process async with asyncio event loop
**Deployment:** Development uses `uvicorn` directly; production should use a process manager like `supervisord` or `systemd`

### WebSocket Endpoint Definition

**Endpoint:** `/ws`
**Protocol:** WebSocket (RFC 6455)
**Subprotocol:** None (binary and text messages intermixed)

**Connection Handshake**
Client initiates WebSocket connection by sending HTTP GET request with `Upgrade: websocket` header. Server accepts and responds with 101 Switching Protocols. Connection is established. Server immediately calls `session_manager.create_session()` and associates the session ID with this WebSocket connection.

**Message Flow After Connection**
Client begins sending video frame messages at 30fps. Server routes each frame to Vision and SLR components. As components produce outputs (refined text from LLM, audio chunks from TTS), server sends them back to client as WebSocket messages. Connection remains open indefinitely until client disconnects or network error occurs.

### Frame Processing Handler

When a binary message arrives (video frame), the orchestrator:

1. Decodes first 8 bytes to extract frame dimensions (height, width)
2. Decodes remaining bytes as BGR frame with those dimensions
3. Increments `session_state.frame_count`
4. If `frame_count % 3 == 0`, submit frame to Vision background thread
5. Submit frame to SLR async queue
6. Return immediately (non-blocking)

### Audio Streaming Handler

When the TTS component yields an audio chunk, the orchestrator:

1. Receives chunk from `tts_queue`
2. Calls `await websocket.send_bytes(chunk)`
3. Updates `session_state.last_audio_timestamp` for latency tracking
4. Continues until all chunks for current sentence are sent

### Error Handling and Disconnection

**Network Errors**
If the WebSocket connection breaks due to network issues, the server catches `WebSocketDisconnect` exception, logs the disconnection reason, and calls `session_manager.destroy_session(session_id)`. All queues are cleared and component processing stops.

**Component Errors**
If any component raises an exception during processing (e.g., Vision model inference fails), the orchestrator catches the exception, logs it, and sends an error message to the client:

```
await websocket.send_json({
    "type": "error",
    "component": "vision",
    "message": "Face detection failed"
})
```

The pipeline continues running with degraded functionality. For example, if Vision fails, TTS defaults to male + neutral voice. If SLR fails, no sign recognition occurs but LLM and TTS remain functional for manually entered text (if that feature is added).

**Graceful Shutdown**
On server shutdown (SIGTERM or SIGINT), the orchestrator:

1. Stops accepting new WebSocket connections
2. Waits for all active sessions to complete their current sentence processing (timeout 10 seconds)
3. Sends disconnect message to all clients
4. Closes all WebSocket connections
5. Shuts down component executors and cleans up resources

### Lifespan Management

FastAPI's lifespan context manager handles startup and shutdown:

```
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.session_manager = SessionManager()
    app.state.vision_engine = VisionInferenceEngine(VisionConfig())
    app.state.vision_engine.load_model(
        gender_head_path="weights/vision/gender_head.pt",
        emotion_head_path="weights/vision/emotion_head.pt"
    )
    app.state.slr_engine = SLRInferenceEngine(SLRConfig())
    app.state.slr_engine.load_model(model_path="weights/slr/slr_model.pt")
    app.state.llm_engine = LLMRefinementEngine(LLMConfig())
    app.state.tts_engine = TTSEngine(TTSConfig())
    
    yield  # Server runs here
    
    # Shutdown
    await app.state.session_manager.destroy_all_sessions()
    # Component cleanup happens automatically via Python destructors
```

### CORS Configuration

For browser clients, CORS headers must be configured:

```
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, replace `"http://localhost:3000"` with the actual frontend domain.

---

## 14. Real-Time Optimization Strategy

Achieving real-time performance requires careful optimization at every layer. This section documents specific techniques applied to each component.

### Vision Component Optimizations

**Frame Stride**
Processing every frame wastes compute — gender and emotion change slowly. Stride of 3 reduces Vision load by 66% with no perceptible quality loss.

**ONNX Quantization**
Export PyTorch model to ONNX INT8 format reduces inference latency by 40-50% on CPU with minimal accuracy degradation. On GPU, FP16 precision achieves 20-30% speedup.

**Perceptual Hash Cache**
Cache hit rate of 60-70% in typical usage (user mostly stationary) eliminates most redundant inferences.

**Background Thread Execution**
Offloading to thread pool prevents blocking the main async loop where SLR and TTS need to run.

### SLR Component Optimizations

**Keypoint Extraction Batching**
MediaPipe Holistic processes one frame at a time, but if queue depth exceeds threshold, batch multiple frames together for amortized overhead.

**Sequence Buffer as Ring Buffer**
Instead of appending and popping from a list (O(n) copy), use a fixed-size NumPy array with a rolling index pointer (O(1) updates).

**Beam Search Pruning**
Reduce beam width from 10 to 5 cuts decoder latency in half with minimal accuracy loss (tested on benchmark dataset).

**Async Keypoint Extraction**
Wrap MediaPipe calls in `asyncio.to_thread` to avoid blocking event loop during extraction.

### LLM Component Optimizations

**Context Window Limiting**
Only include last 5 sentences as context instead of full conversation history. Reduces token count, lowers API costs, and decreases latency.

**Streaming Responses**
Use streaming mode for OpenAI API (`stream=True`) to receive tokens progressively. Start TTS synthesis as soon as first sentence is complete, even if the full response is still generating.

**Local Model Quantization**
For local LLaMA models, use 4-bit GGUF quantization via llama.cpp. Reduces memory footprint from 13GB to 4GB and speeds up inference 3-4×.

**Parallel API Calls**
If multiple sentences are queued, make parallel API requests instead of sequential to reduce total latency.

### TTS Component Optimizations

**Sentence Chunking**
Split long paragraphs into individual sentences and synthesize them in parallel. Stream first sentence audio immediately while second sentence is still synthesizing.

**Voice Model Preloading**
Load all 6 voice models into memory at startup instead of lazy-loading on first use. Eliminates 2-3 second cold start latency.

**Audio Compression**
Use MP3 encoding at 64kbps instead of WAV reduces bandwidth by 85% for network-constrained clients.

**Cache Common Phrases**
Greetings like `"Hello"`, `"Thank you"`, and `"Goodbye"` are cached permanently and returned with zero synthesis latency.

### Network and I/O Optimizations

**WebSocket Binary Framing**
Use binary messages for frames and audio instead of base64-encoded JSON. Reduces payload size by 33%.

**Frame Downsampling**
If client sends 1080p frames, downsample to 480p before processing. SLR and Vision do not benefit from higher resolution but pay the preprocessing cost.

**Asynchronous Logging**
Use asynchronous file writes for logs to avoid blocking the event loop on I/O.

**Connection Pooling**
Reuse HTTP connections for LLM and TTS API calls via `aiohttp.ClientSession` instead of creating new connections per request.

---

## 15. Latency Targets and Performance Benchmarks

Each component has specific latency targets derived from the overall 2.5-second end-to-end goal.

### Vision Component Targets

**GPU Inference**
- Target: 15ms average per frame
- Acceptable: up to 25ms
- Critical threshold: 50ms (causes frame skipping)

**CPU Inference**
- Target: 50ms average per frame
- Acceptable: up to 100ms
- Critical threshold: 150ms

**Measurement Method**
Run `scripts/benchmark_vision.py` with 1000 frames from a test video. Report average, p50, p95, p99 latencies.

### SLR Component Targets

**Keypoint Extraction**
- Target: 8ms average per frame
- Acceptable: up to 15ms

**Sequence Encoding + Decoding**
- Target: 50ms average per 32-frame sequence
- Acceptable: up to 100ms

**Total Per-Frame**
- Target: 60ms average (extraction + encoding)
- Critical threshold: 150ms (causes frame drops)

**Measurement Method**
Run `scripts/benchmark_slr.py` with 1000 frames from a sign language test video. Report per-frame latency.

### LLM Component Targets

**API Call Latency (Cloud)**
- OpenAI GPT-3.5: Target 500ms, Acceptable 1500ms
- OpenAI GPT-4: Target 1000ms, Acceptable 2500ms
- Anthropic Claude: Target 800ms, Acceptable 2000ms

**Local Inference Latency**
- LLaMA 2 7B (quantized): Target 2000ms, Acceptable 5000ms

**Measurement Method**
Run `scripts/benchmark_llm.py` with 100 sample sentences. Report average latency per sentence.

### TTS Component Targets

**Time to First Chunk (Cloud)**
- ElevenLabs: Target 300ms, Acceptable 800ms
- Azure: Target 400ms, Acceptable 1000ms

**Time to First Chunk (Local)**
- Coqui TTS (GPU): Target 800ms, Acceptable 2000ms
- Coqui TTS (CPU): Target 1500ms, Acceptable 3000ms

**Total Synthesis Time (20-word sentence)**
- Cloud: Target 1000ms, Acceptable 2000ms
- Local: Target 2000ms, Acceptable 4000ms

**Measurement Method**
Run `scripts/benchmark_tts.py` with 50 sample sentences of varying lengths. Report time to first chunk and total synthesis time.

### End-to-End Pipeline Targets

**Sign Completion to Audio Start**
- Ideal: 2.0 seconds
- Target: 2.5 seconds
- Acceptable: 3.0 seconds
- Critical: 4.0 seconds (user will perceive lag)

**Breakdown**
- SLR pause detection: 1.0 seconds (configurable)
- LLM refinement: 0.5-1.0 seconds (cloud) or 2.0-3.0 seconds (local)
- TTS first chunk: 0.3-0.8 seconds (cloud) or 1.5-2.0 seconds (local)

**Measurement Method**
Run `tests/integration/test_end_to_end.py` with simulated signing sessions. Measure time from last sign to first audio byte received by client.

---

## 16. Testing and Verification Strategy

Comprehensive testing ensures each component functions correctly in isolation and integrates properly with the full pipeline.

### Unit Testing (Per-Component)

Each component has its own test suite in `tests/test_<component>/` covering:

**Vision Tests**
- Config validation
- Face detection with various inputs (no face, one face, multiple faces)
- Quality filter thresholds
- Transform pipeline output shapes
- Model forward pass shapes and label validity
- Temporal smoother voting logic
- Cache hit/miss behavior
- Inference engine API contract

**SLR Tests**
- Config validation
- Keypoint extraction from frames with visible/occluded hands
- Keypoint normalization correctness
- Sequence buffer FIFO order
- Encoder/decoder output shapes
- Sentence segmentation triggers
- Token confidence thresholding

**LLM Tests**
- Config validation
- Prompt template formatting
- API client initialization (mocked)
- Context window size enforcement
- Retry logic with exponential backoff
- Fallback normalizer activation
- Translation (mocked API calls)

**TTS Tests**
- Config validation
- Voice profile completeness (all 6 profiles defined)
- Voice selector mapping logic
- Text preprocessing (abbreviations, numbers)
- Prosody adjustment parameters
- Cache storage and retrieval
- Synthesizer interface compliance (mocked)

### Integration Testing

Integration tests in `tests/integration/` verify component interactions:

**Vision + SessionState Integration**
Create a mock SessionState, process frames with Vision engine, verify SessionState fields are updated correctly.

**SLR + LLM Integration**
Feed SLR output (raw sign language text) to LLM engine, verify refined output has correct grammar.

**LLM + TTS Integration**
Feed LLM output (refined text) to TTS engine with mock SessionState (gender/emotion), verify correct voice profile is selected.

**Full Pipeline Simulation**
Simulate a complete user session: send video frames, verify Vision updates SessionState, verify SLR recognizes signs, verify LLM refines text, verify TTS synthesizes audio with correct voice.

### End-to-End Testing

The `test_end_to_end.py` test:

1. Starts a test instance of the FastAPI server
2. Connects a WebSocket client
3. Sends a sequence of video frames containing a sign language gesture
4. Waits for TTS audio chunks to arrive
5. Verifies the audio content matches expected output
6. Measures total latency from first frame to first audio chunk
7. Disconnects and shuts down server

### Load and Stress Testing

**Concurrent Sessions Test**
Simulate 10 concurrent WebSocket connections, each sending frames at 30fps. Verify all sessions receive responses without errors and latency remains within acceptable bounds.

**Memory Leak Test**
Run the server for 1 hour with continuous frame processing. Monitor memory usage. Verify no unbounded memory growth occurs.

**Component Failure Test**
Artificially fail each component one at a time (e.g., corrupt model weights, kill API endpoint). Verify the pipeline degrades gracefully without crashing.

### Manual Testing

**Webcam Demos**
`scripts/webcam_demo_vision.py`: Visually verify gender and emotion labels on live webcam feed.

`scripts/webcam_demo_slr.py`: Visually verify recognized sign tokens on live webcam feed.

**Interactive Refinement**
`scripts/test_llm_refinement.py`: Manually enter sign language text and verify LLM refinement quality.

**Voice Synthesis**
`scripts/test_tts_synthesis.py`: Listen to all 6 voice profiles and verify emotional characteristics are appropriate.

---

## 17. Code Quality Standards

All code must adhere to these standards for maintainability and readability.

### Formatting and Style

**Python Version**
All code targets Python 3.10 or 3.11. Use type hints throughout.

**Code Formatter**
Use `black` with default settings (88 character line length). Run `black .` before committing.

**Import Sorting**
Use `isort` with profile `black`. Run `isort .` before committing.

**Linting**
Use `flake8` with `--max-line-length=88` and `--ignore=E203,W503`. All linting errors must be fixed.

### Type Checking

Use `mypy` with strict mode enabled. All functions must have complete type annotations:

```
def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    ...
```

Generic types must be fully specified:

```
from typing import List, Dict, Optional, AsyncGenerator

def get_history() -> List[str]:
    ...

async def stream_audio() -> AsyncGenerator[bytes, None]:
    ...
```

### Docstrings

All public functions, classes, and modules must have docstrings in Google style:

```
def normalize_keypoints(keypoints: np.ndarray, reference_point: tuple) -> np.ndarray:
    """
    Normalize keypoint coordinates relative to a reference point.
    
    Args:
        keypoints: Array of shape (N, 4) with (x, y, z, visibility) columns.
        reference_point: Tuple (x, y, z) for the origin.
    
    Returns:
        Normalized keypoints with same shape as input.
    
    Raises:
        ValueError: If keypoints array has incorrect shape.
    """
```

### Error Handling

**Fail Fast on Critical Errors**
If pretrained weights are missing or config is invalid, raise exceptions immediately during initialization. Do not defer to runtime failures.

**Graceful Degradation on Non-Critical Errors**
If a single frame fails preprocessing, log a warning and return the last stable prediction. Do not crash the pipeline.

**Informative Error Messages**
Include context in all exceptions:

```
raise ValueError(
    f"Expected keypoints shape (N, 4), got {keypoints.shape}. "
    f"Ensure MediaPipe Holistic is returning valid landmarks."
)
```

### Logging

Use `loguru` for all logging. Configure log levels:

- DEBUG: Frame-by-frame processing details (disabled in production)
- INFO: Component initialization, session creation, major events
- WARNING: Degraded functionality (e.g., cache miss, API retry)
- ERROR: Component failures that do not crash the pipeline
- CRITICAL: Fatal errors that prevent the pipeline from functioning

Include context in all log messages:

```
logger.info(
    "Vision prediction updated",
    session_id=session_state.session_id,
    gender=result["gender"],
    emotion=result["emotion"],
    confidence=result["emotion_conf"]
)
```

### File Organization

**Single Responsibility**
Each file should have one clear purpose. Do not mix unrelated functionality.

**Small Files**
Aim for files under 300 lines. If a file exceeds 500 lines, refactor into smaller modules.

**Logical Grouping**
Group related functions and classes in the same file. For example, all face detection logic in `face_detector.py`, all gender/emotion heads in `heads.py`.

---

## 18. Implementation Order and Dependencies

This section specifies the exact order in which files and functions should be implemented to minimize dependency errors.

### Phase 1: Foundation (Days 1-2)

**Day 1**
1. Create directory structure per Section 3
2. Create all `__init__.py` files (initially empty)
3. Write `setup.py` with package metadata and dependencies
4. Create `requirements-cpu.txt` and `requirements-gpu.txt`
5. Write `shared/session_state.py` defining SessionState dataclass
6. Write `shared/message_types.py` defining WebSocket message schemas
7. Write `shared/logging_config.py` configuring Loguru

**Day 2**
8. Write `vision/config.py` defining VisionConfig dataclass
9. Write `slr/config.py` defining SLRConfig dataclass
10. Write `llm/config.py` defining LLMConfig dataclass
11. Write `tts/config.py` defining TTSConfig dataclass
12. Write `scripts/verify_installation.py` to check all imports
13. Test: Run verify script and ensure all imports succeed

### Phase 2: Vision Component (Days 3-6)

**Day 3**
14. Implement `vision/preprocessing/face_detector.py` — FaceDetector class using MediaPipe
15. Implement `vision/preprocessing/transforms.py` — NumPyToTensorTransform and normalization
16. Implement `vision/preprocessing/quality_filter.py` — blur and brightness assessment
17. Write `tests/test_vision/test_preprocessing.py`
18. Test: Run `pytest tests/test_vision/test_preprocessing.py -v`

**Day 4**
19. Implement `vision/models/backbone.py` — ViTBackbone loading from transformers
20. Implement `vision/models/heads.py` — GenderClassificationHead and EmotionClassificationHead
21. Implement `vision/models/vision_model.py` — MultiTaskVisionModel combining backbone and heads
22. Write `tests/test_vision/test_model.py`
23. Test: Run `pytest tests/test_vision/test_model.py -v`

**Day 5**
24. Implement `vision/inference/temporal_smoother.py` — TemporalSmoother and PairedSmoother
25. Implement `vision/inference/cache.py` — PerceptualHasher and PredictionCache
26. Write `tests/test_vision/test_smoother.py` and `tests/test_vision/test_cache.py`
27. Test: Run smoother and cache tests

**Day 6**
28. Implement `vision/inference/engine.py` — VisionInferenceEngine public API
29. Update `vision/__init__.py` to export VisionConfig and VisionInferenceEngine
30. Write `tests/test_vision/test_engine.py`
31. Write `scripts/benchmark_vision.py` and `scripts/webcam_demo_vision.py`
32. Test: Run all Vision tests and verify benchmarks

### Phase 3: SLR Component (Days 7-11)

**Day 7**
33. Implement `slr/preprocessing/keypoint_normalizer.py` — coordinate normalization
34. Implement `slr/preprocessing/sequence_buffer.py` — sliding window buffer
35. Write `tests/test_slr/test_keypoint_extraction.py` and `tests/test_slr/test_sequence_buffer.py`
36. Test: Run preprocessing tests

**Day 8**
37. Implement `slr/models/pose_extractor.py` — MediaPipe Holistic wrapper
38. Implement `slr/utils/vocabulary.py` — sign token vocabulary mappings
39. Write basic tests for pose extraction

**Day 9**
40. Implement `slr/models/sequence_encoder.py` — LSTM or Transformer encoder
41. Implement `slr/inference/token_decoder.py` — beam search decoder
42. Write tests for encoder and decoder output shapes

**Day 10**
43. Implement `slr/inference/sentence_segmenter.py` — pause and punctuation detection
44. Implement `slr/models/slr_model.py` — complete SLRModel assembling all pieces

**Day 11**
45. Implement `slr/inference/engine.py` — SLRInferenceEngine public API
46. Update `slr/__init__.py` to export SLRConfig and SLRInferenceEngine
47. Write `tests/test_slr/test_engine.py`
48. Write `scripts/benchmark_slr.py` and `scripts/webcam_demo_slr.py`
49. Test: Run all SLR tests and verify benchmarks

### Phase 4: LLM Component (Days 12-14)

**Day 12**
50. Implement `llm/clients/openai_client.py` — OpenAI API wrapper
51. Implement `llm/clients/anthropic_client.py` — Anthropic API wrapper
52. Implement `llm/clients/local_client.py` — local LLaMA wrapper
53. Write `tests/test_llm/test_clients.py` with mocked API responses

**Day 13**
54. Implement `llm/prompts/refinement_prompts.py` — system and user prompts
55. Implement `llm/prompts/translation_prompts.py` — translation prompts
56. Implement `llm/processing/text_normalizer.py` — basic normalization
57. Implement `llm/processing/context_manager.py` — conversation context tracking
58. Write tests for prompts and processing modules

**Day 14**
59. Implement `llm/processing/fallback_handler.py` — retry logic
60. Implement `llm/inference/engine.py` — LLMRefinementEngine public API
61. Update `llm/__init__.py` to export LLMConfig and LLMRefinementEngine
62. Write `tests/test_llm/test_engine.py`
63. Write `scripts/test_llm_refinement.py` for interactive testing
64. Test: Run all LLM tests

### Phase 5: TTS Component (Days 15-18)

**Day 15**
65. Implement `tts/voices/voice_profiles.py` — define 6 voice profiles
66. Implement `tts/voices/voice_selector.py` — selection logic
67. Write tests for voice profile completeness and selector

**Day 16**
68. Implement `tts/synthesis/synthesizer_interface.py` — abstract base class
69. Implement `tts/synthesis/coqui_synthesizer.py` — Coqui TTS wrapper
70. Implement `tts/synthesis/elevenlabs_synthesizer.py` — ElevenLabs API wrapper
71. Implement `tts/synthesis/azure_synthesizer.py` — Azure TTS wrapper
72. Write tests for synthesizers (mocked)

**Day 17**
73. Implement `tts/processing/text_preprocessor.py` — abbreviation and number expansion
74. Implement `tts/processing/prosody_adjuster.py` — emotion-based adjustments
75. Implement `tts/processing/audio_chunker.py` — sentence splitting
76. Write tests for processing modules

**Day 18**
77. Implement `tts/inference/engine.py` — TTSEngine public API
78. Update `tts/__init__.py` to export TTSConfig and TTSEngine
79. Write `tests/test_tts/test_engine.py`
80. Write `scripts/test_tts_synthesis.py` and `scripts/benchmark_tts.py`
81. Test: Run all TTS tests

### Phase 6: Orchestrator (Days 19-21)

**Day 19**
82. Implement `orchestrator/session_manager.py` — SessionState lifecycle management
83. Implement `orchestrator/frame_router.py` — dispatch frames to Vision and SLR
84. Write `tests/test_orchestrator/test_session_manager.py`
85. Test: Run session manager tests

**Day 20**
86. Implement `orchestrator/audio_streamer.py` — stream TTS chunks to WebSocket
87. Implement `orchestrator/websocket_handler.py` — WebSocket route handlers
88. Implement `orchestrator/app.py` — FastAPI app setup with lifespan
89. Write `tests/test_orchestrator/test_websocket_handler.py`

**Day 21**
90. Update `orchestrator/__init__.py` to export OrchestratorApp
91. Write `scripts/run_server.py` to start the server
92. Test: Start server and connect with a WebSocket test client
93. Verify basic frame routing works

### Phase 7: Integration and End-to-End (Days 22-24)

**Day 22**
94. Write `tests/integration/test_end_to_end.py`
95. Test: Run end-to-end test and verify all components integrate correctly
96. Fix any integration bugs discovered

**Day 23**
97. Run all benchmark scripts and record latencies
98. Optimize any components exceeding latency targets
99. Re-run benchmarks and verify improvements

**Day 24**
100. Write README.md with setup and usage instructions
101. Review and update all docstrings
102. Run `black`, `isort`, `flake8`, `mypy` on entire codebase
103. Fix all style and type errors
104. Run `pytest tests/ -v --cov` and verify >90% coverage
105. Tag release v1.0.0

---

## 19. Edge Cases and Error Handling Requirements

### Vision Component Edge Cases

**No Face Detected**
If MediaPipe Face Detection returns an empty bounding box, the inference engine should return a result with `data_quality="no_face"` and reuse the last stable prediction. After 90 consecutive no-face frames, reset the temporal smoother to avoid stale predictions.

**Multiple Faces Detected**
If multiple faces are detected, use the largest face by bounding box area. Log a warning if more than one face is detected consistently (may indicate multiple people on camera).

**Face Too Small**
If the detected face bounding box is smaller than `FACE_MIN_SIZE` pixels on either dimension, reject the frame as low quality and return last stable prediction.

**Model Produces NaN or Inf**
After forward pass, check logits with `torch.isfinite(logits).all()`. If non-finite values are found, log a warning with input tensor statistics and return last stable prediction.

**Pretrained Weights Not Found**
If `weights/vision/gender_head.pt` or `weights/vision/emotion_head.pt` is missing, raise `RuntimeError` with clear message indicating which file is missing and where to place it. Do not proceed with uninitialized weights.

**ONNX Model Not Found**
If `use_onnx=True` in config but ONNX file path is invalid or file does not exist, fall back to PyTorch inference and log a warning.

### SLR Component Edge Cases

**Hand Occlusion**
If MediaPipe Holistic returns landmarks with visibility scores below 0.3 for more than 50% of hand keypoints, interpolate missing keypoints from adjacent frames. If interpolation is not possible (first frame or long occlusion), skip the frame and do not add to sequence buffer.

**Sequence Buffer Underflow**
If the sequence buffer has fewer than 32 frames (e.g., at session start), pad the buffer with zeros or duplicate the first frame until buffer is full. Accept that predictions may be less accurate until full buffer is populated.

**Decoder Outputs No Token**
If beam search returns all probabilities below `TOKEN_CONFIDENCE_THRESHOLD`, do not append any token to sentence buffer. This filters out non-sign movements.

**Sentence Buffer Overflow**
If sentence buffer exceeds `MAX_SENTENCE_TOKENS` (default 50), force segmentation and emit the buffered sentence even if no explicit boundary was detected. Clear buffer and continue.

**Queue Depth Exceeded**
If SLR async queue depth exceeds `QUEUE_DEPTH_LIMIT`, drop the oldest frame in queue and log a warning. This prevents unbounded memory growth when SLR cannot keep up with frame rate.

### LLM Component Edge Cases

**API Call Timeout**
If LLM API call exceeds 10 seconds, cancel the request and retry with exponential backoff. After 3 retries, fall back to the rule-based normalizer.

**API Rate Limit Hit**
If API returns HTTP 429 (Too Many Requests), parse the `Retry-After` header and wait the specified duration before retrying. If no header, use exponential backoff.

**API Returns Empty Response**
If API returns an empty string or only whitespace, log an error and fall back to the rule-based normalizer for this sentence.

**API Returns Malformed JSON**
If API response cannot be parsed as JSON (for structured output modes), log an error and retry once. If retry fails, fall back to normalizer.

**Translation to Unsupported Language**
If `TARGET_LANGUAGE` is set to a language code not supported by the LLM API, log a warning and skip translation. Return the English refined text only.

**Context Window Overflow**
If the conversation context grows beyond the LLM's token limit (e.g., 4096 tokens for GPT-3.5), truncate the oldest sentences from context until under limit. Log a warning if truncation occurs frequently.

### TTS Component Edge Cases

**Voice Profile Not Found**
If the combination of (gender, emotion) in SessionState does not match any defined voice profile, fall back to male + neutral and log a warning.

**Synthesizer Initialization Failure**
If Coqui TTS model fails to load (e.g., missing weight files), attempt to fall back to cloud API if credentials are configured. If no fallback is available, raise an error and disable TTS for this session.

**API Call Timeout**
If TTS API call exceeds 5 seconds, retry once. If retry fails, log an error and return silence (empty audio chunk) rather than blocking the pipeline.

**Audio Encoding Failure**
If `pydub` fails to encode audio to MP3 (e.g., missing ffmpeg), fall back to WAV format and log a warning.

**Empty Text Input**
If the text string passed to TTS is empty or only whitespace, skip synthesis and return immediately without error. Do not send empty audio chunks.

**Text Contains Unsupported Characters**
If the text contains non-ASCII characters not supported by the voice model, attempt to transliterate or strip the characters and log a warning. If text becomes empty after cleaning, skip synthesis.

### Orchestrator Edge Cases

**WebSocket Disconnection Mid-Frame**
If the WebSocket connection drops while a frame is being transmitted, catch `WebSocketDisconnect` exception and destroy the session gracefully without crashing other sessions.

**Component Raises Unexpected Exception**
If any component raises an exception not explicitly handled, catch it in the orchestrator, log the full traceback, send an error message to the client, and continue processing. Do not crash the server.

**Memory Leak Detection**
If memory usage exceeds 4GB per session (unrealistic for normal use), log a critical error and forcibly disconnect that session to prevent OOM crashes.

**Session Creation Failure**
If SessionState creation fails (e.g., out of memory), reject the WebSocket connection with HTTP 503 Service Unavailable and log the error.

**Component Initialization Failure on Startup**
If any component fails to initialize during the lifespan startup phase (e.g., missing weights), log the error and raise an exception to prevent the server from starting. Do not allow the server to accept connections if components are not functional.

---

## 20. Completion Checklist

Use this checklist to verify the implementation is complete and correct before declaring the project done.

### Structure and Setup
- [ ] All directories from Section 3 exist with correct structure
- [ ] All `__init__.py` files are in place and export appropriate symbols
- [ ] `setup.py` allows `pip install -e .` to succeed
- [ ] `requirements-cpu.txt` and `requirements-gpu.txt` contain all pinned versions
- [ ] `.env.example` lists all required API keys
- [ ] `.gitignore` excludes `weights/`, `logs/`, `.env`, `__pycache__`
- [ ] `scripts/verify_installation.py` passes without errors
- [ ] `weights/` and `logs/` directories exist with `.gitkeep` files

### Configuration
- [ ] All four config dataclasses (Vision, SLR, LLM, TTS) instantiate without errors
- [ ] All config fields have correct types and valid default values
- [ ] Config tests in each component's test directory pass

### Vision Component
- [ ] Face detector returns empty list for blank frame
- [ ] Face detector returns single face for frame with one person
- [ ] Face alignment produces correct output shape (96, 96, 3)
- [ ] Quality filter correctly rejects blurry frames (variance < threshold)
- [ ] Quality filter correctly rejects dark frames (brightness < threshold)
- [ ] Transform pipeline produces tensor with shape (1, 3, 96, 96)
- [ ] ViT backbone loads from transformers library without errors
- [ ] All backbone parameters have `requires_grad=False`
- [ ] Gender head: input (1, 384) → output (1, 2)
- [ ] Emotion head: input (1, 384) → output (1, 3)
- [ ] Model `predict` method returns valid labels and confidences
- [ ] Temporal smoother majority voting works correctly
- [ ] Cache returns hits for identical face crops
- [ ] Inference engine processes frames within latency target
- [ ] All Vision tests pass: `pytest tests/test_vision/ -v`
- [ ] Webcam demo displays correct predictions: `python scripts/webcam_demo_vision.py`
- [ ] Benchmark meets targets: `python scripts/benchmark_vision.py`

### SLR Component
- [ ] Keypoint extractor returns correct shape (543, 4) for valid frame
- [ ] Keypoint normalizer centers coordinates on shoulder midpoint
- [ ] Sequence buffer maintains FIFO order and correct size
- [ ] Encoder forward pass produces correct output shape
- [ ] Decoder beam search outputs valid token indices
- [ ] Sentence segmentation triggers on pause threshold
- [ ] Sentence segmentation triggers on punctuation gesture
- [ ] Confidence thresholding rejects low-probability tokens
- [ ] All SLR tests pass: `pytest tests/test_slr/ -v`
- [ ] Webcam demo displays recognized signs: `python scripts/webcam_demo_slr.py`
- [ ] Benchmark meets targets: `python scripts/benchmark_slr.py`

### LLM Component
- [ ] OpenAI client initializes correctly with API key
- [ ] Anthropic client initializes correctly with API key
- [ ] Local client initializes correctly with model weights
- [ ] Refinement prompts format correctly with sample inputs
- [ ] Translation prompts format correctly with target language
- [ ] Context manager maintains sliding window of last 5 sentences
- [ ] Retry logic executes 3 attempts with exponential backoff
- [ ] Fallback normalizer activates after max retries
- [ ] All LLM tests pass: `pytest tests/test_llm/ -v`
- [ ] Interactive test script refines text correctly: `python scripts/test_llm_refinement.py`
- [ ] Benchmark meets targets: `python scripts/benchmark_llm.py`

### TTS Component
- [ ] All 6 voice profiles defined (male/female × sad/neutral/happy)
- [ ] Voice selector maps (gender, emotion) to correct profile
- [ ] Voice selector defaults to male + neutral for invalid inputs
- [ ] Text preprocessor expands abbreviations correctly
- [ ] Text preprocessor normalizes numbers to words
- [ ] Prosody adjuster applies correct parameters for each emotion
- [ ] Coqui synthesizer loads voice model without errors
- [ ] ElevenLabs synthesizer makes API calls successfully (if credentials configured)
- [ ] Azure synthesizer makes API calls successfully (if credentials configured)
- [ ] Cache stores and retrieves audio correctly
- [ ] All TTS tests pass: `pytest tests/test_tts/ -v`
- [ ] Synthesis test generates all 6 voices: `python scripts/test_tts_synthesis.py`
- [ ] Benchmark meets targets: `python scripts/benchmark_tts.py`

### Orchestrator
- [ ] Session manager creates and destroys sessions correctly
- [ ] Frame router dispatches frames to Vision and SLR queues
- [ ] Audio streamer sends TTS chunks to WebSocket
- [ ] WebSocket handler accepts connections and routes messages
- [ ] FastAPI app starts without errors: `python scripts/run_server.py`
- [ ] Lifespan context manager initializes all components on startup
- [ ] Lifespan context manager cleans up on shutdown
- [ ] All orchestrator tests pass: `pytest tests/test_orchestrator/ -v`

### Integration and End-to-End
- [ ] End-to-end test passes: `pytest tests/integration/test_end_to_end.py -v`
- [ ] Server handles concurrent WebSocket connections without errors
- [ ] Memory usage remains stable over 1-hour test
- [ ] Component failure tests demonstrate graceful degradation
- [ ] Browser client connects and receives audio successfully

### Code Quality
- [ ] All code formatted with `black`: `black .`
- [ ] All imports sorted with `isort`: `isort .`
- [ ] All linting passes: `flake8 . --max-line-length=88 --ignore=E203,W503`
- [ ] All type checks pass: `mypy .`
- [ ] All public functions have docstrings
- [ ] All tests have descriptive names and assertions

### Documentation
- [ ] README.md provides clear setup and usage instructions
- [ ] `.env.example` lists all required API keys with descriptions
- [ ] All component READMEs (if present) are up to date
- [ ] IMPLEMENTATION_GUIDE.md reflects final architecture

### Performance
- [ ] Vision latency within targets (Section 15)
- [ ] SLR latency within targets (Section 15)
- [ ] LLM latency within targets (Section 15)
- [ ] TTS latency within targets (Section 15)
- [ ] End-to-end pipeline latency under 2.5 seconds (ideal) or 3.0 seconds (acceptable)

### Final Verification
- [ ] Run full test suite: `pytest tests/ -v --cov`
- [ ] Test coverage >90%
- [ ] No failing tests
- [ ] No critical or error-level log messages during normal operation
- [ ] Demo session recorded and verified working end-to-end
- [ ] Project ready for deployment

---

*End of Comprehensive Implementation Guide*

---

**Document version:** 3.0 (Complete Pipeline)  
**Components:** Vision (Gender + Emotion Detection), SLR (Sign Language Recognition), LLM (Text Refinement), TTS (Voice Synthesis)  
**Part of:** Real-Time AI Meeting Assistant Pipeline  
**Intended reader:** CLI-based LLM executing implementation steps sequentially
