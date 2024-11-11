from flask import Flask, request, render_template, jsonify
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from PIL import Image
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Create 'static/uploads' directory if it doesn't exist
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Load model and processor once
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
classifier_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file)

    # Convert image to RGB to ensure compatibility
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Process image
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        logits = classifier_model(**inputs).logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()
    label = classifier_model.config.id2label[predicted_label]
    
    # Save the image so that it can be shown in the HTML
    image_path = 'static/uploads/image.jpg'
    image.save(image_path)
    
    # Return JSON response with label and image URL
    return jsonify({
        "label": label,
        "image_url": f"/static/uploads/image.jpg"  # Ensure correct image URL path
    })


# Function to get embedding for a word
def get_word_embedding(word):
    embedding = embedding_model.encode([word])
    return embedding.tolist()

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5
    return dot_product / (norm1 * norm2)

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    data = request.get_json()
    word = data['word']
    embedding = get_word_embedding(word)
    return jsonify({'embedding': embedding})

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    word1 = data['word1']
    word2 = data['word2']
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    similarity = cosine_similarity(embedding1[0], embedding2[0])
    return jsonify({'similarity': similarity})


if __name__ == '__main__':
    app.run(debug=True)






