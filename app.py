from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe (jeśli używasz .env lokalnie)
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_TOKEN or not MODEL_NAME:
    raise ValueError("Brak wymaganych zmiennych środowiskowych: API_TOKEN lub MODEL_NAME")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Brak wiadomości w żądaniu"}), 400

        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"error": "Pusta wiadomość"}), 400

        prompt = f"""Odpowiadaj na pytania użytkownika w sposób rzetelny, dokładny i zgodny z nauką. Unikaj fałszywych informacji. 

Użytkownik: {user_input}
Asystent:"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.5,  # Obniżona kreatywność
                "max_new_tokens": 200
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)

        if response.status_code != 200:
            return jsonify({
                "error": f"Błąd API Hugging Face: {response.status_code}",
                "details": response.text
            }), response.status_code

        chatbot_response = response.json()

        if isinstance(chatbot_response, list) and len(chatbot_response) > 0:
            generated_text = chatbot_response[0].get("generated_text", "").strip()
            clean_response = generated_text.replace(prompt, "").strip() if generated_text else "Brak odpowiedzi"
        else:
            clean_response = "Brak odpowiedzi lub niepoprawny format odpowiedzi"

        return jsonify({"response": clean_response})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Błąd podczas komunikacji z API Hugging Face", "details": str(e)}), 500

    except Exception as e:
        return jsonify({"error": "Wewnętrzny błąd serwera", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
