from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
try:
    # Próbujemy pobrać listę baz danych
    print(client.list_database_names())
    print("Połączenie z MongoDB udane!")
except Exception as e:
    print("Błąd połączenia:", e)
