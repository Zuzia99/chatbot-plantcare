import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

# Wymuszamy działanie na CPU
device = torch.device("cpu")

# Ładowanie modelu i tokenizera
print("Ładuję model GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")  # Dodajemy device_map="cpu"
print("Model GPT-2 załadowany!")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        # Tokenizacja wejścia, jawnie przenosimy tensory na CPU
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Upewniamy się, że wszystko jest na CPU
        inputs = {key: value.to(device) for key, value in inputs.items()}  

        # Generowanie odpowiedzi na CPU
        with torch.no_grad():  # Zoptymalizowane pod kątem wydajności
            output = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
