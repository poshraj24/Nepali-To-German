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
nepali_sentence = "म खाना खानछु "
translate_and_print(nepali_sentence)