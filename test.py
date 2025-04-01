from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Załaduj model i tokenizer
print("Ładowanie modelu...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sprawdzenie, czy model i tokenizer zostały załadowane
print("Model i tokenizer załadowane.")

# Tokenizacja wejścia
inputs = tokenizer("Cześć!", return_tensors="pt")
print(f"Tokenized input: {inputs}")

# Generowanie odpowiedzi
outputs = model.generate(inputs['input_ids'], max_length=50)
print(f"Generated output: {outputs}")

# Dekodowanie odpowiedzi
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Odpowiedź modelu: {response}")
