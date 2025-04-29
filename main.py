import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as gen_ai
import pyttsx3
import speech_recognition as sr
import fitz  # PyMuPDF
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import threading

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-2.0-flash')

# Initialize text to speech engine
engine = pyttsx3.init()
def speak_text(text):
    engine.say(text)
    engine.runAndWait()
def speak_text_in_thread(text):
    threading.Thread(target=speak_text, args=(text,)).start()

#OCR arabic reader
ocr_reader = easyocr.Reader(['ar'])

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
knowledge_chunks = []
knowledge_embeddings = None
faiss_index = None

# Define a system prompt for guiding the AI responses
SYSTEM_PROMPT = """
أنت مساعد ذكي اسمه "أرابيكو"، متخصص فقط في تحليل النصوص العربية المرفوعة من ملفات PDF أو صور.
يجب أن تكون جميع ردودك باللغة العربية فقط وان تقوم بترجمتها الى اللغة الانجليزية، ولا تعتمد على معلومات خارج النص المعالج.
كن مهذبًا، ودودًا، واحترافيًا، وركز فقط على تحليل وفهم النصوص. ابدا المحادثة برسالة ترحيبية تعرف بخدماتك 
"""
# Function to handle the chunk of text
def chunk_text(text, max_words=100):
    sentences = text.split('.')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= max_words:
            chunk += sentence + "."
        else:
            chunks.append(chunk.strip())
            chunk = sentence + "."
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def prepare_knowledge_base(text):
    global knowledge_chunks, knowledge_embeddings, faiss_index
    knowledge_chunks = chunk_text(text)
    knowledge_embeddings = embedding_model.encode(knowledge_chunks)
    faiss_index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
    faiss_index.add(knowledge_embeddings)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf, filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def extract_text_from_image(image_bytes):
    result = ocr_reader.readtext(image_bytes)
    return " ".join([item[1] for item in result])

def search_knowledge_base(query, top_k=1):
    if not faiss_index:
        return ""
    query_vector = embedding_model.encode([query])
    D, I = faiss_index.search(query_vector, top_k)
    return " ".join([knowledge_chunks[i] for i in I[0]])

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.text(" تحدث الآن...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="ar-SA")
    except:
        return ""

# Function to translate roles between Gemini and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.chat_session.send_message(SYSTEM_PROMPT)

# Streamlit UI
st.set_page_config(page_title="Chat with Arabico", layout="centered")
st.title("🤖 Arabico - AI-Powered Arabic Content Analysis Agent")
st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .chat-row {
            display: flex;
            justify-content: space-between;
            width: 100%;
        }
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            word-wrap: break-word;
            display: inline-block;
        }
        .user-message {
            background-color: #ffd8cc;
            text-align: right;
        }
        .bot-message {
            background-color: #ffd8cc;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# Chat history container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_session.history[1:]:
    role = translate_role_for_streamlit(message.role)
    css_class = "user-message" if role == "user" else "bot-message"
    position = "flex-end" if role == "user" else "flex-start"
    
    st.markdown(f'<div class="chat-row" style="justify-content: {position};">'
                f'<div class="chat-message {css_class}">{message.parts[0].text}</div>'
                f'</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# File uploader for images and PDFs
uploaded_file = st.file_uploader("Upload an image /file for analysis", type=["pdf", "jpg", "jpeg", "png"])
if uploaded_file:
    file_type = uploaded_file.type
    with st.spinner(" استخراج النص..."):
        if file_type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file.read())
        else:
            extracted_text = extract_text_from_image(uploaded_file.getvalue())
        prepare_knowledge_base(extracted_text)
    st.success(" تم تجهيز قاعدة المعرفة من الملف")

user_prompt = st.chat_input(" Ask Arabico or Speak ..")

if user_prompt and faiss_index:
        relevant_text = search_knowledge_base(user_prompt)
        gemini_response = model.generate_content(SYSTEM_PROMPT + "\n\n" + relevant_text + "\n\nالسؤال: " + user_prompt)
        st.markdown(f" {gemini_response.text}")
        if st.button(" استمع للإجابة"):
            speak_text_in_thread( gemini_response.text)

if st.button(" تحدث بدلاً من الكتابة"):
    voice_text = transcribe_audio()
    if voice_text:
        st.text(f" انت قلت: {voice_text}")
        relevant_text = search_knowledge_base(voice_text)
        gemini_response = model.generate_content(SYSTEM_PROMPT + "\n\n" + relevant_text + "\n\nالسؤال: " + voice_text)
        st.markdown(f"{gemini_response.text}")
        speak_text_in_thread( gemini_response.text)

# Add welcome message
if "welcome_message_shown" not in st.session_state:
    st.session_state.welcome_message_shown = True
    st.markdown(f'<div class="chat-row" style="justify-content: flex-start;">'
                f'</div>', unsafe_allow_html=True)
    engine.say("مرحبًا بك! كيف يمكنني مساعدتك اليوم؟")
    engine.runAndWait()

if user_prompt:
    st.markdown(f'<div class="chat-row" style="justify-content: flex-end;">'
                f'<div class="chat-message user-message">{user_prompt}</div>'
                f'</div>', unsafe_allow_html=True)
    
    gemini_response = st.session_state.chat_session.send_message(user_prompt)
    
    st.markdown(f'<div class="chat-row" style="justify-content: flex-start;">'
                f'<div class="chat-message bot-message">{gemini_response.text}</div>'
                f'</div>', unsafe_allow_html=True)
    engine.say(gemini_response.text)
    engine.runAndWait()

