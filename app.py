import io

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(
    torch.load("resnet50_retinamnist.pth", map_location=torch.device('cpu'))
)
model.eval()


@app.route("/", methods=["GET"])
def health():
    return "Welcome to the RetinaMNIST API!", 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        image = transform(image).unsqueeze(0)  # Add batch dimension
        model.eval()
        with torch.no_grad():
            output = model(image)
            print(output)
        _, predicted = torch.max(output, 1)
        return jsonify({'predicted_label': predicted.item()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
