import torch
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved translation pipeline
translation_pipeline = torch.load('translation_pipeline.pth', map_location=device)

# Make sure the loaded pipeline models are on the correct device
translation_pipeline.nep_model.to(device)
translation_pipeline.ger_model.to(device)

def translate_and_print(text):
    # Perform Nepali to English translation
    english_translation = translation_pipeline.nep_2_eng(text)
    # Perform English to German translation
    german_translation = translation_pipeline.nep_2_ger(english_translation)
    
    print(f"Nepali: {text}")
    print(f"English: {english_translation}")
    print(f"German: {german_translation}")

# Example usage
#nepali_sentence = "नेपालमा आज भोलि धेरै सवारी दुर्घटनाहरु भैराखेका छन् "
from google.colab import files
uploaded = files.upload()
import speech_recognition as sr

r = sr.Recognizer()
audio_file = list(uploaded.keys())[0]
with sr.AudioFile(audio_file) as source:
    audio_text = r.listen(source)
    try:
        text = r.recognize_google(audio_text,language="ne")
        print("Converting audio transcripts into text ...")
        print(text)
    except sr.RequestError:
        print("Speech recognition service is currently unavailable.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")

#a=int(input("Enter the number of translations you would like to continue:"))
#for _ in range(a):  
user_input = text
translate_and_print(user_input)