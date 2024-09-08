import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, FSMTForConditionalGeneration, FSMTTokenizer, pipeline, BartForConditionalGeneration, BartTokenizer
import speech_recognition as sr
from PIL import Image
import streamlit as st
from gtts import gTTS
from io import BytesIO
import base64
import cohere # Import the cohere

# Set the path to your OpenAI API key
# cohere_api_key = 'yssuUFNKvpOXrsms21oHNVoUUOH0RYYHPBPvzZpl'
# co = cohere.Client(cohere_api_key)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models and tokenizers once to avoid repeated loading
nep_2_eng_model_name = 'facebook/nllb-200-distilled-600M'
nep_tokenizer = AutoTokenizer.from_pretrained(nep_2_eng_model_name, src_lang="npi_Deva")
nep_model = AutoModelForSeq2SeqLM.from_pretrained(nep_2_eng_model_name).to(device)

eng_2_ger_model_name = 'facebook/wmt19-en-de'
ger_tokenizer = FSMTTokenizer.from_pretrained(eng_2_ger_model_name)
ger_model = FSMTForConditionalGeneration.from_pretrained(eng_2_ger_model_name).to(device)

class TranslationPipeline:
    def __init__(self, nep_model, nep_tokenizer, ger_model, ger_tokenizer):
        self.nep_model = nep_model
        self.nep_tokenizer = nep_tokenizer
        self.ger_model = ger_model
        self.ger_tokenizer = ger_tokenizer

    def nep_2_eng(self, text):
        inputs = self.nep_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = self.nep_model.generate(**inputs, forced_bos_token_id=self.nep_tokenizer.convert_tokens_to_ids("eng_Latn"))
        eng_translation = self.nep_tokenizer.decode(translated[0], skip_special_tokens=True)
        return eng_translation

    def eng_2_ger(self, text):
        inputs = self.ger_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = self.ger_model.generate(**inputs)
        ger_translation = self.ger_tokenizer.decode(translated[0], skip_special_tokens=True)
        return ger_translation

    def nep_2_ger(self, text):
        eng_text = self.nep_2_eng(text)
        ger_text = self.eng_2_ger(eng_text)
        return ger_text

# Initialize the translation pipeline
translation_pipeline = TranslationPipeline(nep_model, nep_tokenizer, ger_model, ger_tokenizer)

# Function to recognize speech and return the text
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        st.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Please Speak Now...")
        audio = recognizer.listen(source)
        st.info("Recognizing speech...Please Wait")
        
    try:
        response = recognizer.recognize_google(audio, language="ne")
        return response
    except sr.RequestError:
        st.error("API unavailable")
        return None
    except sr.UnknownValueError:
        st.error("Unable to recognize speech")
        return None

# Function to convert text to speech and play it
def text_to_speech(text, lang="de"):
    tts = gTTS(text=text, lang=lang)
    tts_fp = BytesIO()
    tts.write_to_fp(tts_fp)
    tts_fp.seek(0)
    return tts_fp

# Function to generate autoplay audio HTML
def generate_autoplay_audio_html(audio_bytes):
    # Encode the audio file in base64
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    return audio_html

# Function to generate similar questions using GPT-4
def generate_similar_questions(text):
    # Load the BART model and tokenizer from Hugging Face
    model_name = "facebook/bart-large-cnn"  # You can also explore other BART variants
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Prepare the prompt for BART
    prompt = f"Generate three similar questions based on the input: '{text}'."
    
    # Encode the input text and generate questions
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=100, num_beams=4, early_stopping=True)
    
    # Decode the generated text and return
    questions = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    return questions

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Streamlit UI
image = Image.open('logo.PNG')
st.image(image)

# Global variables
title = "Nepali to German Translation"
app_dsc = "Your App to translate the language from <i>Nepali</i>, TO <i>German</i>"

# Description Section 
st.markdown("<h1 style='text-align: center;'>"+title+"</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>"+app_dsc+"</h3>", unsafe_allow_html=True)

# Start the translation process
if st.button("Start Voice Input"):
    with st.spinner("Listening and Translating..."):
        recognized_text = recognize_speech_from_mic(recognizer, microphone)
        if recognized_text:
            st.success(f"You said: {recognized_text}")
            german_translation = translation_pipeline.nep_2_ger(recognized_text)
            st.write(f"Translation to German: {german_translation}")
            
            # Convert the German text to speech
            audio_fp = text_to_speech(german_translation, lang="de")
            audio_bytes = audio_fp.read()

            # Generate autoplay audio HTML and embed it in Streamlit
            audio_html = generate_autoplay_audio_html(audio_bytes)
            st.markdown(audio_html, unsafe_allow_html=True)
            
            # Generate similar questions based on recognized text
            similar_questions = generate_similar_questions(german_translation)
            st.write("Talking Buddy:")
            st.write(similar_questions)
            audio_fp1=text_to_speech(similar_questions, lang="de")
            audio_bytes1 = audio_fp1.read()
            audio_html1 = generate_autoplay_audio_html(audio_bytes1)
            st.markdown(audio_html1, unsafe_allow_html=True)

        else:
            st.warning("No speech detected, please try again.")

st.stop()  # Stop the Streamlit app until the button is clicked again
