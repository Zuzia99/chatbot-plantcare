import requests

# Twój API token z Hugging Face
API_TOKEN = 'aaa' # Zastąp tym tokenem, który uzyskałaś w kroku 2

# Model GPT, np. "gpt2" lub inne dostępne modele na Hugging Face
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Ustawienie modelu na papuGaPT2
# MODEL_NAME = "flax-community/papuGaPT2"

# Adres URL API Hugging Face
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Nagłówki z tokenem autoryzacyjnym
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Przykładowe zapytanie
data = {
    "inputs": "Jesteś ekspertem od roślin. Pisz po polsku. Jak podlewać kaktusy?",
    "parameters": {
        "temperature": 0.5,  # Zwiększenie kreatywności, ale z zachowaniem precyzji
        "max_length": 50,  # Zwiększenie długości odpowiedzi
    }
}

# Wysyłamy zapytanie do modelu
response = requests.post(API_URL, headers=headers, json=data)

# Wyświetlamy odpowiedź
if response.status_code == 200:
    result = response.json()
    print("Odpowiedź chatbota:", result[0]['generated_text'])
else:
    print(f"Błąd: {response.status_code}, {response.text}")
