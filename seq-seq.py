import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, FSMTForConditionalGeneration, FSMTTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TranslationPipeline:
    def __init__(self):
        # Load Nepali-to-English NLLB model and tokenizer
        nep_2_eng_model_name = 'facebook/nllb-200-distilled-600M'
        self.nep_tokenizer = AutoTokenizer.from_pretrained(nep_2_eng_model_name, src_lang="npi_Deva")
        self.nep_model = AutoModelForSeq2SeqLM.from_pretrained(nep_2_eng_model_name).to(device)
        
        # Load English-to-German FSMT model and tokenizer
        eng_2_ger_model_name = 'facebook/wmt19-en-de'
        self.ger_tokenizer = FSMTTokenizer.from_pretrained(eng_2_ger_model_name)
        self.ger_model = FSMTForConditionalGeneration.from_pretrained(eng_2_ger_model_name).to(device)

    def nep_2_eng(self, text):
        # Tokenize and move inputs to the device (GPU)
        inputs = self.nep_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate English translation from Nepali
        translated = self.nep_model.generate(**inputs, forced_bos_token_id=self.nep_tokenizer.convert_tokens_to_ids("eng_Latn"))

        # Decode the translated text
        eng_translation = self.nep_tokenizer.decode(translated[0], skip_special_tokens=True)
        return eng_translation

    def eng_2_ger(self, text):
        # Tokenize and move inputs to the device (GPU)
        inputs = self.ger_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate German translation from English
        translated = self.ger_model.generate(**inputs)

        # Decode the translated text
        ger_translation = self.ger_tokenizer.decode(translated[0], skip_special_tokens=True)
        return ger_translation

    def nep_2_ger(self, text):
        eng_text = self.nep_2_eng(text)
        ger_text = self.eng_2_ger(eng_text)
        return ger_text

# Initialize the translation pipeline
translation_pipeline = TranslationPipeline()

# Save the model (note: saving the model here saves the class, which is not typical. Usually, you save models separately.)
torch.save(translation_pipeline, 'translation_pipeline.pth')