!pip install sacremoses
!pip install datasets
import sacremoses
from transformers import MarianMTModel, MarianTokenizer, FSMTForConditionalGeneration, FSMTTokenizer
from datasets import load_dataset

ds = load_dataset("ashokpoudel/nepali-english-translation-dataset")

#load nep-eng model and tokenizer
nep_2_eng_model_name = 'Helsinki-NLP/opus-mt-ne-en'
nep_tokenizer = MarianTokenizer.from_pretrained(ds)
nep_model = MarianMTModel.from_pretrained(nep_2_eng_model_name)

#load eng-german model and tokenizer
eng_2_ger_model_name = 'facebook/wmt19-en-de'
ger_tokenizer = FSMTTokenizer.from_pretrained(eng_2_ger_model_name)
ger_model = FSMTForConditionalGeneration.from_pretrained(eng_2_ger_model_name)

#translation operation
def nep_2_eng(text):
    inputs = nep_tokenizer(text, return_tensors="pt", padding=True, truncation = True)
    #english translation from nepali
    translated = nep_model.generate(**inputs)

    #decode the translated text
    eng_translation = nep_tokenizer.decode(translated[0], skip_special_tokens=True)
    return eng_translation

def eng_2_ger(text):
    inputs = ger_tokenizer(text, return_tensors="pt", padding=True, truncation = True)
    #german translation from english
    translated = ger_model.generate(**inputs)

    #decode the translated text
    ger_translation = ger_tokenizer.decode(translated[0], skip_special_tokens=True)
    return ger_translation

def nep_2_ger(text):
    eng_text = nep_2_eng(text)
    ger_text = eng_2_ger(eng_text)
    return ger_text

nepali_sentence = "केरा"
print(f"Nepali: {nepali_sentence}")
english_translation= nep_2_eng(nepali_sentence)
print(f"English: {english_translation}")
german_translation = eng_2_ger(english_translation)
print(f"German: {german_translation}")


