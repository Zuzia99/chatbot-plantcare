require("dotenv").config();
console.log("🔍 URI z .env:", process.env.MONGO_URI); // Sprawdzenie wartości

// Import wymaganych bibliotek
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3000;

// Połączenie z MongoDB
mongoose
  .connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("✅ Połączono z MongoDB"))
  .catch((err) => console.error("❌ Błąd połączenia z MongoDB:", err));

// Middleware
app.use(express.json());
app.use(cors()); // Pozwala frontendowi komunikować się z backendem

// Prosty endpoint testowy
app.get("/", (req, res) => {
  res.send("Chatbot działa! 🌱");
});

// Endpoint dla chatbota
app.post('/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: "Brak wiadomości w zapytaniu!" });
    }

    let response = "";

    // Sprawdzamy zapytanie o podlewanie
    if (message.toLowerCase().includes("podlewanie")) {
        response = "Podlewanie zależy od gatunku rośliny. Na przykład sukulentów podlewamy rzadziej, a paprocie potrzebują więcej wilgoci.";
    }
    // Sprawdzamy zapytanie o słońce
    else if (message.toLowerCase().includes("słońce")) {
        response = "Większość roślin lubi jasne miejsce, ale bez bezpośredniego słońca. Wyjątkiem są kaktusy, które preferują pełne słońce.";
    }
    // Zapytanie o nawożenie
    else if (message.toLowerCase().includes("nawożenie")) {
        response = "Rośliny doniczkowe powinny być nawożone od wiosny do lata co 2-4 tygodnie. Zimą nie jest to konieczne.";
    }
    // Jeśli brak dopasowania
    else {
        response = "Podlewanie roślin zależy od ich gatunku.";
    }

    res.json({ response });
});


// Uruchamiamy serwer
app.listen(PORT, () => {
  console.log(`🚀 Serwer działa na http://localhost:${PORT}`);
});

