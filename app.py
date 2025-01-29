from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# Créer un dossier pour stocker les fichiers uploadés
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fonction d'entraînement du modèle (très simplifiée pour l'exemple)
def train_model(dataset_path):
    # Charger le tokenizer GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Charger le modèle GPT-2
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Charger et préparer le dataset
    with open(dataset_path, 'r') as f:
        text = f.read()
    
    # Tokenizer le texte
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

    # Entraîner le modèle (exemple très simplifié, l'entraînement réel est plus complexe)
    model.train()
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    # Sauvegarder le modèle et le tokenizer
    model.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')

    return 'trained_model/pytorch_model.bin', 'trained_model/tokenizer_config.json'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier téléchargé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier non valide'}), 400
    
    # Sauvegarder le fichier téléchargé
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Entraîner le modèle avec le dataset
    model_file, tokenizer_file = train_model(filepath)

    # Retourner les fichiers .pth et tokenizer.json (ici, on peut juste renvoyer des liens vers les fichiers)
    return jsonify({
        'model_file': model_file,
        'tokenizer_file': tokenizer_file
    })

if __name__ == '__main__':
    app.run(debug=True)
