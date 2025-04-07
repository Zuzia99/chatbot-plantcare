from flask import Flask, render_template, request, jsonify, g
import requests
from flask_cors import CORS
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
import traceback

# Załaduj zmienne środowiskowe (jeśli używasz .env lokalnie)
load_dotenv()

# Tworzenie aplikacji Flask
app = Flask(__name__)
CORS(app)

# Ustawienie loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sprawdzenie zmiennej MONGODB_URI
MONGODB_URI = os.getenv("MONGODB_URI")
print("MONGODB_URI:", MONGODB_URI)

# Zmienne API
API_TOKEN = os.getenv("API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_TOKEN or not MODEL_NAME:
    raise ValueError("Brak wymaganych zmiennych środowiskowych: API_TOKEN lub MODEL_NAME")
if not MONGODB_URI:
    raise ValueError("Brak zmiennej środowiskowej: MONGODB_URI")

# Funkcja lazy initialization dla MongoDB
def get_db():
    if 'db' not in g:
        try:
            g.db = MongoClient(MONGODB_URI).get_database()
        except Exception as e:
            logger.error(f"Błąd połączenia z MongoDB: {e}")
            raise e
    return g.db

# Zamknięcie połączenia przy końcu kontekstu aplikacji
@app.teardown_appcontext
def close_db(error=None):
    db = g.pop('db', None)
    if db is not None:
        db.client.close()

# Konfiguracja API Hugging Face
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}?task=text-generation"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Domyślny routing do strony głównej
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        logger.info("Otrzymano żądanie POST /chat")
        data = request.get_json()
        logger.info(f"Otrzymane dane JSON: {data}")
        
        if not data or "message" not in data:
            return jsonify({"error": "Brak wiadomości w żądaniu"}), 400

        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"error": "Pusta wiadomość"}), 400

        # Uproszczony prompt systemowy – jednoznaczna instrukcja dla modelu
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Jesteś ekspertem ogrodniczym. Udzielaj krótkich, jednozdaniowych, konkretnych i rzeczowych odpowiedzi na pytania dotyczące pielęgnacji roślin. "
                        "Twoja odpowiedź powinna zaczynać się od słowa 'Odpowiedź:' i nie zawierać powtórzeń pytania ani dodatkowych informacji."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            "parameters": {
                "temperature": 0.1,
                "max_new_tokens": 60,
                "repetition_penalty": 2.0
                # Usuwamy parametr "stop", żeby model sam zakończył generację
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        logger.info(f"Status response: {response.status_code}")
        logger.info(f"Response text: {response.text}")
        
        if response.status_code != 200:
            return jsonify({
                "error": f"Błąd API Hugging Face: {response.status_code}",
                "details": response.text
            }), response.status_code

        chatbot_response = response.json()
        logger.info(f"Pełna odpowiedź modelu: {chatbot_response}")

        # Pobieramy wygenerowany tekst
        if isinstance(chatbot_response, dict) and "generated_text" in chatbot_response:
            generated_text = chatbot_response["generated_text"].strip()
        elif isinstance(chatbot_response, list) and len(chatbot_response) > 0:
            generated_text = chatbot_response[0].get("generated_text", "").strip()
        else:
            generated_text = "Brak odpowiedzi"

        # Jeśli wygenerowany tekst zawiera marker "Odpowiedź:", wycinamy wszystko przed nim
        if "Odpowiedź:" in generated_text:
            clean_response = generated_text.split("Odpowiedź:")[1].strip()
        else:
            clean_response = generated_text.replace(user_input, "").strip()

        try:
            chats_collection = get_db().chats
            chats_collection.insert_one({"user_input": user_input, "bot_response": clean_response})
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania do bazy danych MongoDB: {e}")
            return jsonify({"error": "Błąd zapisu do bazy danych"}), 500

        return jsonify({"response": clean_response})

    except requests.exceptions.RequestException as e:
        logger.error("Błąd podczas komunikacji z API Hugging Face: " + str(e))
        return jsonify({"error": "Błąd podczas komunikacji z API Hugging Face", "details": str(e)}), 500

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error("Błąd serwera: " + error_details)
        return jsonify({"error": "Wewnętrzny błąd serwera", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
