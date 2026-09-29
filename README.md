# CodeAlpha Artificial Intelligence Internship

**Intern Name:** Mann Patel  
**Domain:** Artificial Intelligence  
**Duration:** December 2025  
**Organization:** CodeAlpha  

---

## 📌 Project Overview

This repository contains all four comprehensive, production-grade tasks developed during the **CodeAlpha Artificial Intelligence Internship**. The projects span across key areas of modern Artificial Intelligence, including **Natural Language Processing (NLP)**, **Conversational AI / Information Retrieval**, **Deep Learning Generative Sequence Modeling (LSTM)**, and **Computer Vision / Real-time Object Tracking (YOLOv8)**.

---

## 📋 Task Checklist & Status

- [x] **Task 1: Language Translation Tool (`GlobalConnect AI`)** — Includes Text-to-Speech (TTS), Document Translation & Language Swap.
- [x] **Task 2: Intelligent FAQ Chatbot (`TechHub AI Assistant`)** — Includes Smart Intent Handling, Quick Chips, Confidence Badges & Chat Export.
- [x] **Task 3: AI Music Generation with LSTM (`NeuroMelody AI`)** — Includes Web Studio, Tempo/Creativity Controls & MIDI Exporter.
- [x] **Task 4: Real-time Object Detection & Tracking (`YOLOv8 Pro`)** — Includes Live Analytics HUD, Snapshot Capture & Browser Vision Studio.

---

## 🛠️ Tasks Breakdown & Enhanced Features

### 🌐 Task 1: Enterprise Neural Translator (`GlobalConnect AI`)
- **Directory:** [`Task1_LanguageTranslation/`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task1_LanguageTranslation/)
- **Core Script:** [`Task1_LanguageTranslation/app.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task1_LanguageTranslation/app.py)
- **Tech Stack:** Python, Gradio, `deep-translator`, `gTTS`
- **✨ Enhanced Features:**
  - **Live Neural Translation:** Auto-detects source language across 25+ global and Indian regional languages.
  - **Text-to-Speech (TTS):** Instant audio generation and playback for translated outputs.
  - **Language Swap:** Quick button to swap source and target languages.
  - **Real-time Analytics:** Word counter, character counter, and status badges.
  - **Batch File Translation Tab:** Upload `.txt` documents to translate and download results.
- **Run Command:**
  ```bash
  python Task1_LanguageTranslation/app.py
  ```

---

### 💬 Task 2: Intelligent Customer Support Chatbot (`TechHub AI Assistant`)
- **Directory:** [`Task2_FAQChatbot/`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task2_FAQChatbot/)
- **Core Scripts:** [`Task2_FAQChatbot/app.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task2_FAQChatbot/app.py), [`Task2_FAQChatbot/faq_data.csv`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task2_FAQChatbot/faq_data.csv)
- **Tech Stack:** Python, Gradio, Scikit-learn, NLTK, Pandas
- **✨ Enhanced Features:**
  - **TF-IDF & N-Gram Semantics:** Calculates cosine similarity for accurate query resolution.
  - **Smart Intent Routing:** Built-in recognition of greetings, gratitude, and goodbyes.
  - **Confidence Metrics:** Displays confidence match percentages on responses.
  - **Quick Suggestion Chips:** 1-click buttons for common questions (Orders, Warranty, EMI, Returns).
  - **Transcript Export:** Download the conversation history as a text file.
- **Run Command:**
  ```bash
  python Task2_FAQChatbot/app.py
  ```

---

### 🎹 Task 3: Generative Music Composition (`NeuroMelody AI`)
- **Directory:** [`Task3_MusicGeneration/`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task3_MusicGeneration/)
- **Core Scripts:** [`Task3_MusicGeneration/app.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task3_MusicGeneration/app.py) (Web Studio), [`Task3_MusicGeneration/music_ai.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task3_MusicGeneration/music_ai.py) (Training), [`Task3_MusicGeneration/generate.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task3_MusicGeneration/generate.py) (Inference)
- **Tech Stack:** Python, TensorFlow / Keras (LSTM), `music21`, NumPy, Gradio
- **✨ Enhanced Features:**
  - **Deep LSTM Sequence Model:** 2-layer stacked LSTM with Dropout (0.3) trained on polyphonic MIDI notes and chords.
  - **Interactive Web Studio:** Control note sequence count (30-300), tempo (Allegro/Moderato/Andante), and instrument selection (Piano, Electric Piano, Guitar, Violin, Flute).
  - **Temperature / Creativity Tuning:** Adjust temperature sampling for predictable or expressive musical variance.
  - **Export to MIDI:** Generates standard `.mid` files playable on any media player or DAW.
- **Run Commands:**
  ```bash
  # Launch Interactive Music Studio Web App:
  python Task3_MusicGeneration/app.py

  # Train / Re-train Model:
  python Task3_MusicGeneration/music_ai.py
  ```

---

### 🎥 Task 4: Real-time Object Detection & Tracking (`YOLOv8 Pro`)
- **Directory:** [`Task4_ObjectDetection/`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task4_ObjectDetection/)
- **Core Scripts:** [`Task4_ObjectDetection/tracker.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task4_ObjectDetection/tracker.py) (Live OpenCV), [`Task4_ObjectDetection/app.py`](file:///c:/Users/mann2/OneDrive/Desktop/CodeAlpha_Artificial_Intelligence_Internship/Task4_ObjectDetection/app.py) (Browser Studio)
- **Tech Stack:** Python, Ultralytics YOLOv8 Nano, OpenCV (`cv2`), Gradio
- **✨ Enhanced Features:**
  - **Real-Time Analytics HUD:** Live FPS counter, active track ID count, and per-class object breakdown banner.
  - **Interactive Stream Controls:** Hotkeys for Pause/Resume (`P`), Snapshot (`S`), and HUD Toggle (`H`).
  - **Snapshot Manager:** Automatically saves timestamped detection frames into `Task4_ObjectDetection/captures/`.
  - **Web Vision Studio:** Upload images to adjust Confidence and IoU thresholds interactively.
- **Run Commands:**
  ```bash
  # Live Webcam / Video Tracker with HUD:
  python Task4_ObjectDetection/tracker.py

  # Browser-based Vision Studio:
  python Task4_ObjectDetection/app.py
  ```

---

## 💻 Environment Setup & Installation

1. **Activate Virtual Environment:**
   - **PowerShell (Windows):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Command Prompt (Windows):**
     ```cmd
     venv\Scripts\activate.bat
     ```

2. **Install Unified Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Repository Structure

```
CodeAlpha_Artificial_Intelligence_Internship/
├── Task1_LanguageTranslation/
│   ├── app.py                      # Enhanced Translator (TTS, File Translation, Swap)
│   └── requirements.text
├── Task2_FAQChatbot/
│   ├── app.py                      # Enhanced FAQ Chatbot (Chips, Confidence, Export)
│   └── faq_data.csv                # E-commerce FAQ dataset
├── Task3_MusicGeneration/
│   ├── app.py                      # Interactive Music Studio Web UI
│   ├── music_ai.py                 # MIDI preprocessing & LSTM training pipeline
│   ├── generate.py                 # Neural music composition with Temperature sampling
│   └── music_data/                 # MIDI training dataset
├── Task4_ObjectDetection/
│   ├── tracker.py                  # Live YOLOv8 Tracker with Analytics HUD & Snapshot
│   ├── app.py                      # Browser-based YOLOv8 Vision Studio
│   └── captures/                   # Saved detection snapshots
├── INTERNSHIP_DETAILS.txt          # In-depth internship report and architecture notes
├── README.md                       # Main internship documentation
├── requirements.txt                # Unified requirements file
├── ai_generated_song.mid           # AI generated MIDI song output
├── music_brain.h5                  # Trained LSTM neural network weights
└── yolov8n.pt                      # YOLOv8 pre-trained model weights
```

---

## 👤 Author
**Mann Patel**  
Artificial Intelligence Intern — **CodeAlpha** (December 2025)
