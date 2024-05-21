import torch
import urllib.request
from io import BytesIO
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont

public_url = 'https://storage.googleapis.com/dataset-wasteapp/50epoch_rebalance.pth'

def load_model_from_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                model_state_dict = torch.load(BytesIO(response.read()), map_location=torch.device('cpu'))
                return model_state_dict
            else:
                raise Exception(f"Failed to download file. Status code: {response.status}")
    except urllib.error.URLError as e:
        raise Exception(f"Failed to download file. Error: {e.reason}")

# Load the state dict from the URL
checkpoint = load_model_from_url(public_url)

# Define the classes
class_names = [
    'background',
    'Botol Kaca',
    'Botol Plastik',
    'Galon',
    'Gelas Plastik',
    'Kaleng',
    'Kantong Plastik',
    'Kantong Semen',
    'Kardus',
    'Kemasan Plastik',
    'Kertas Bekas',
    'Koran',
    'Pecahan Kaca',
    'Toples Kaca',
    'Tutup Galon'
]

# Extract the model weights from the 'model_state_dict' key
model_weights = checkpoint['model_state_dict']

# Initialize the model with the number of classes it was trained with
num_classes = len(class_names)  # Number of classes including background
model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(model_weights)
model.eval()

def get_prediction(image_bytes, threshold=0.5):
    img = Image.open(image_bytes).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    selected_boxes = []
    selected_labels = []
    selected_scores = []

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            selected_boxes.append(box)
            selected_labels.append(class_names[label])
            selected_scores.append(score.item())

    return img, selected_boxes, selected_labels, selected_scores

def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=4)
        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(((x1, y1 - (text_bbox[3] - text_bbox[1])), (x1 + (text_bbox[2] - text_bbox[0]), y1)), fill="yellow")
        draw.text((x1, y1 - (text_bbox[3] - text_bbox[1])), text, fill="black", font=font)

    return image
