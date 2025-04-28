import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as gen_ai
import pyttsx3
import speech_recognition as sr
import fitz  # PyMuPDF
import threading

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Arabico",
    page_icon=":robot:",
    layout="centered",  
)

# Set up Google Gemini-Pro AI model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)

model = gen_ai.GenerativeModel('gemini-2.0-flash')

def extract_text_from_pdf(uploaded_pdf):
    # فتح الملف PDF
    doc = fitz.open(stream=uploaded_pdf, filetype="pdf")
    
    text = ""
    # استخراج النص من جميع الصفحات
    for page in doc:
        text += page.get_text()

    return text

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to handle the text-to-speech process in a separate thread
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to use threading for speaking text without blocking the main thread
def speak_text_in_thread(text):
    # Create a new thread to handle the text-to-speech process
    thread = threading.Thread(target=speak_text, args=(text,))
    thread.start()

# Function to transcribe audio to text
def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.text("أرجو التحدث...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language="ar-SA")  # Using Arabic language model
        st.text(f"أنت قلت: {text}")
        return text
    except sr.UnknownValueError:
        st.text("لم أتمكن من فهم الصوت.")
        return ""
    except sr.RequestError as e:
        st.text(f"حدث خطأ في الاتصال بخدمة التعرف على الصوت: {e}")
        return ""

# Function to handle the user's audio input and response
def handle_audio_input():
    # Record the audio
    text = transcribe_audio()
    if text:
        # Send text to GeminiAI
        gemini_response = model.generate_content(text)
        st.markdown(f"{gemini_response.text}")
        # Speak the bot's response
        speak_text(gemini_response.text)
        
# Define a system prompt for guiding the AI responses
SYSTEM_PROMPT = """{
  "input": {
    "user": {
      "greeting": "مرحبًا! كيف يمكنني مساعدتك اليوم؟",
      "intent": "تحليل النصوص"
    },
    "documents": [
      {"type": "pdf", "file": "دروس اللغة العربية لغير الناطقين بها.pdf","الركيزة في اصول التفسير.pdf"},
      {"type": "image", "file": "Image.jpg"},
      {"type": "audio", "file": "manspeakingarabic.mp3"}
    ]
  },
  "output": {
    "actions": [
      {"action": "extract_text_from_pdf"},
      {"action": "extract_text_from_image", "language": "arabic"},
      {"action": "process_voice_input", "language":"arabic"}
    ],
    "filter": {
      "question_topic": "تحليل النصوص"
    },
    "response_format": {
      "text": {
        "language": "arabic",
        "translation": "english"
      },
      "audio": {
        "language": "arabic",
        "translation": "english"
      }
    }
  },
  "system_prompt": "أنت نموذج لغوي مخصص لتحليل النصوص باللغة العربية فقط. يجب أن ترد فقط باللغة العربية ولا تستخدم أي لغة أخرى. يجب عليك أن تبدأ المحادثة بالترحيب اللطيف مع ذكر اسمك أرابيكو، ثم تسأل المستخدم عن مراده. بعد ذلك، تلتزم فقط بالرد على الأسئلة المتعلقة بتحليل النصوص والملفات المرفوعة لك والتي تدربت عليها. كما يجب عليك أن تترجم جميع ردودك إلى اللغة الإنجليزية."
}
"""

# Function to translate roles between Gemini and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.chat_session.send_message(SYSTEM_PROMPT)


# Display the chatbot's title on the page
st.title("🤖Arabico - AI-Powered Arabic Content Analysis Agent")
st.markdown(
    """
    <style>
        .main {
            background-color:!important;
        }
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
            max-width: 45%;
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
    """,
    unsafe_allow_html=True
)

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

# File uploader for images
uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg", "pdf"])
if uploaded_file:
    # Check if the uploaded file is a PDF
    if uploaded_file.type == "application/pdf":
        # Extract text from the PDF file
        pdf_text = extract_text_from_pdf(uploaded_file.read())
        st.markdown(f'<div class="chat-row" style="justify-content: flex-start;">'
                    f'<div class="chat-message bot-message">{pdf_text}</div>'
                    f'</div>', unsafe_allow_html=True)
        
        gemini_response = st.session_state.chat_session.send_message(pdf_text)
        st.markdown(f'<div class="chat-row" style="justify-content: flex-start;">'
                    f'<div class="chat-message bot-message">{gemini_response.text}</div>'
                    f'</div>', unsafe_allow_html=True)
    else:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(image, width=200, caption="Uploaded Image")
        
    gemini_response = st.session_state.chat_session.send_message([image])
    st.markdown(f'<div class="chat-row" style="justify-content: flex-start;">'
                f'<div class="chat-message bot-message">{gemini_response.text}</div>'
                f'</div>', unsafe_allow_html=True)
    
    uploaded_file = None

# Input field for user's message
user_prompt = st.chat_input("Ask Arabico...")

# Add welcome message and hide initial prompt
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

st.markdown("### اضغط على الزر لتسجيل الصوت أو أدخل النص مباشرة.")

# Button to trigger audio input
if st.button("تفعيل إدخال الصوت"):
    handle_audio_input()  # Handle audio input when the button is pressed
