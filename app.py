import io
import base64
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
MODEL = load_model("./best_mnist_model.keras")
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Recognizer</title>
    <style>
        body { display: flex; flex-direction: column; align-items: center; margin-top: 50px; font-family: Arial, sans-serif; }
        #canvas { border: 2px solid #000; touch-action: none; }
        .controls { margin-top: 10px; }
        button { padding: 10px 20px; font-size: 16px; margin: 0 5px; }
        #prediction { font-size: 24px; margin-top: 20px; }
    </style>
</head>
<body>
    <canvas id="canvas" width="280" height="280"></canvas>
    <div class="controls">
        <button id="clear">Clear</button>
        <button id="predict">Predict</button>
    </div>
    <div id="prediction">Draw a digit and click "Predict"</div>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        let drawing = false;
        let lastPos = null;

        canvas.addEventListener('pointerdown', e => {
            drawing = true;
            lastPos = { x: e.offsetX, y: e.offsetY };
        });
        canvas.addEventListener('pointermove', e => {
            if (!drawing) return;
            const pos = { x: e.offsetX, y: e.offsetY };
            ctx.beginPath();
            ctx.moveTo(lastPos.x, lastPos.y);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            lastPos = pos;
        });
        canvas.addEventListener('pointerup', () => drawing = false);
        document.getElementById('clear').onclick = () => {
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('prediction').innerText = 'Draw a digit and click "Predict"';
        };
        document.getElementById('predict').onclick = () => {
            const dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('prediction').innerText = 'Prediction: ' + data.label;
            });
        };
    </script>
</body>
</html>
'''  

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        raw = base64.b64decode(img_data)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        # Resize to 28x28
        resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        inp = norm.reshape(1, 28, 28, 1)
        pred = MODEL.predict(inp)
        label = LABELS[int(np.argmax(pred))]
        return jsonify({'label': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
