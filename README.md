# 🧠 AI-Powered Arabic Content Analysis Agent

A smart assistant application that analyzes Arabic text from PDF documents and images using Retrieval-Augmented Generation (RAG) techniques, and responds through both text and voice.

## 📋 Features

- Upload and analyze Arabic PDFs and images.
- Extract text from Arabic images using OCR.
- Chunk documents into manageable pieces.
- Generate multilingual embeddings.
- Store and retrieve information intelligently using a vector database (FAISS).
- Natural conversation via text input/output.
- Voice interaction (speech recognition + text-to-speech).
- Translation from Arabic to English for responses.


## 🛠 Tech Stack

- **Python**
- **Streamlit** - Fast web interface development
- **PyMuPDF (fitz)** - PDF processing
- **EasyOCR** - OCR for Arabic images
- **SentenceTransformers** - Embedding generation
- **FAISS** - Vector database for efficient retrieval
- **Google Generative AI (Gemini)** - Content generation
- **pyttsx3** - Local text-to-speech
- **speech_recognition** - Voice-to-text conversion
- **Threading** - Non-blocking audio playback
- **dotenv** - Manage environment variables securely



## 🔍 Difference Between `main.py` and `app.py`

The project contains two main scripts:

| File | Purpose | Description |
|------|---------|-------------|
| `app.py` | Initial Prototype | This file was used to test basic functionalities like uploading PDFs, generating responses, and simple voice interaction. It does **not** include OCR for images, chunking, or vector-based retrieval. |
| `main.py` | Final Version | This is the complete version that follows the full RAG framework. It supports Arabic OCR, text chunking, embedding generation using SentenceTransformers, smart retrieval with FAISS, and a more robust text+voice interaction system. |

➤ The `main.py` version is the one that fully meets the requirements of the technical assessment and should be used for final deployment or demonstration.


## 🚀 Installation & Usage

1. **Download** or **extract** the ZIP folder of the project.
2. Open a terminal in the project root and install dependencies:
    pip install -r requirements.txt
    
3. Create a `.env` file and add your Google API Key:
    
    GOOGLE_API_KEY=your_google_api_key_here
    
4. Run the application:
    streamlit run main.py or streamlit run app.py
    

⚡ **Note**: Due to `pyttsx3` dependency on external programs like `eSpeak`, full functionality (especially voice responses) is guaranteed only on **local environments**.


## ⚡ Challenges Overcome

- Implemented Arabic OCR extraction effectively using EasyOCR.
- Designed text chunking to handle long documents properly.
- Integrated FAISS to enable smart retrieval of content.
- Managed non-blocking text-to-speech functionality with threading.
- Local deployment recommended due to `pyttsx3` dependency issues with cloud platforms.
