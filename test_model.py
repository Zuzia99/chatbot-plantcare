import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Rozpoczynam ładowanie modelu...")

# Używamy use_fast=False, aby uniknąć problemów z SentencePiece
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", torch_dtype=torch.float16, device_map="auto")

print("Model załadowany!")
input("Naciśnij Enter, aby zakończyć...")
