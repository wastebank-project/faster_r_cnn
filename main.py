from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
from model import get_prediction, draw_boxes

app = Flask(__name__)

@app.route("/")
def main():
    return "Response Successful!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image_format = Image.open(image_bytes).format  # Detect image format
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Get predictions
    img, boxes, labels, scores = get_prediction(image_bytes, threshold=0.7)
    img_with_boxes = draw_boxes(img, boxes, labels, scores)

    # Save the image with boxes to a BytesIO object
    img_io = io.BytesIO()
    img_with_boxes.save(img_io, format=image_format)  # Save in the original format
    img_io.seek(0)

    return send_file(img_io, mimetype=f'image/{image_format.lower()}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)