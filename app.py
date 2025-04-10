from flask import Flask, render_template, request, jsonify, g
import requests
from flask_cors import CORS
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
import traceback
import html  # do konwersji kodów HTML
import time

# Załaduj zmienne środowiskowe
load_dotenv()

app = Flask(__name__)
CORS(app)

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

logging.info(f"Uruchamiam model: {MODEL_NAME}")

def get_db():
    if 'db' not in g:
        try:
            g.db = MongoClient(MONGODB_URI).get_database()
        except Exception as e:
            logger.error(f"Błąd połączenia z MongoDB: {e}")
            raise e
    return g.db

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

def send_request_with_retry(payload, headers, retries=5, delay=3):
    retries = int(retries)
    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    
    for attempt in range(retries):
        try:
            logger.info(f"Wysyłam zapytanie: próba {attempt+1}/{retries} na URL: {url}")
            logger.info(f"Payload: {payload}")
            response = requests.post(url, json=payload, headers=headers)
            logger.info(f"Odpowiedź (próba {attempt+1}): Status: {response.status_code} - Treść: {response.text[:200]}")
            
            if response.status_code == 503:
                logger.error("Błąd 503: Usługa jest chwilowo niedostępna. Spróbuj ponownie później.")
                return {"error": "Serwis jest chwilowo niedostępny. Spróbuj później."}
            
            response.raise_for_status()
            logger.info("Odpowiedź API została pomyślnie odebrana.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Próba {attempt+1}/{retries} nie powiodła się: {e}")
            if attempt < retries - 1:
                logger.info(f"Ponawiam próbę za {delay} sekund.")
                time.sleep(delay)
            else:
                logger.error("API Hugging Face nie odpowiada po maksymalnej liczbie prób.")
                return {"error": "API Hugging Face nie odpowiada. Spróbuj później."}

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
        
        # Zmodyfikowany prompt
        prompt = (
            "Instrukcje: Jesteś ekspertem ogrodniczym. Odpowiadaj na pytania wyłącznie po polsku, w sposób naturalny, przyjazny i zwięzły. "
            "Twoja odpowiedź musi zaczynać się od słowa 'Odp:' i nie może zawierać żadnych dodatkowych informacji ani powtórzeń pytania. "
            "Upewnij się, że Twoja odpowiedź jest logiczna, pełna i zakończona pełnymi zdaniami.\n"
            f"Pytanie: {user_input}\n"
            "=====\n"
            "Odp:"
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.3,      # Wyższa temperatura może dać większą różnorodność
                "max_new_tokens": 150,   # Zwiększamy limit tokenów, aby odpowiedź była pełna
                "repetition_penalty": 1.5,
                "stop_sequence": "\n"    # Kończymy generowanie przy znaku nowej linii
            }
        }

        response = send_request_with_retry(payload, HEADERS)

        # Przetwarzanie odpowiedzi
        if isinstance(response, dict) and "generated_text" in response:
            generated_text = response["generated_text"].strip()
        elif isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "").strip()
        else:
            generated_text = "Nie działam jeszcze idealnie... jestem w trakcie metamorfozy. ;-) Zapraszam ponownie później."

        # Użycie html.unescape() do konwersji kodów HTML
        generated_text = html.unescape(generated_text)

        # Wyodrębnienie czystej odpowiedzi - szukamy części po "Answer:"
        if "Answer:" in generated_text:
            clean_response = generated_text.split("Answer:", 1)[1].strip()
        else:
            clean_response = generated_text

        # Zamiana pozostałych encji HTML
        clean_response = clean_response.replace("&#039;", "'").replace("&quot;", "\"")

        # Zapis do bazy danych
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
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
