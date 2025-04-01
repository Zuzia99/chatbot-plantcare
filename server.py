import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

print("Ładuję model...")
# Używamy use_fast=False, by uniknąć problemów z SentencePiece (jeśli występują)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", torch_dtype=torch.float16, device_map="auto")
print("Model załadowany!")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"response": "Nie podałeś żadnej wiadomości!"})
    
    # Generowanie odpowiedzi
    inputs = tokenizer(message, return_tensors="pt")
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is None:
        attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=attention_mask,  # Dodaj attention_mask
        max_length=200
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
