import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ustaw token z pliku .env lub bezpośrednio
hf_token = os.getenv('HF_TOKEN')

# Załaduj model i tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", torch_dtype=torch.float16, device_map="auto")


# Testowanie - generowanie odpowiedzi na przykładowe zapytanie
input_text = "Jak pielęgnować rośliny doniczkowe?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'])

# Dekodowanie i wypisanie odpowiedzi
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Dodaj komunikat po załadowaniu modelu
print("Model załadowany!")
input("Naciśnij Enter, aby zakończyć...")