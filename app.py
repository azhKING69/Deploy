import io
import base64
from flask import Flask, request, jsonify
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

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

        # Prepare image
        pil_img = Image.open(io.BytesIO(raw)).convert('L')
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)

        # Prepare input tensor
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        input_shape = input_details['shape']
        arr = arr.reshape(input_shape)

        interpreter.set_tensor(input_details['index'], arr)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details['index'])

        label = LABELS[int(np.argmax(pred))]
        return jsonify({'label': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
