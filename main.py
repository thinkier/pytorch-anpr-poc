from ultralytics import YOLO
lpr = YOLO("yolov8x-supervision-license-plate-recognition best.pt")

import torch
ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
from strhub.data.module import SceneTextDataModule
img_transform = SceneTextDataModule.get_transform(ocr.hparams.img_size)

import io
from PIL import Image
with open("in/image.jpg", "rb") as f:
    image_buffer = io.BytesIO(f.read())
image = Image.open(image_buffer).convert("RGB")

plates = lpr.predict(image, save=True, show_labels=True)

for plate in plates[0].boxes.xyxy:
    x1, y1, x2, y2 = plate.tolist()

    image_plate = image.crop((x1, y1, x2, y2))
    logits = ocr(img_transform(image_plate).unsqueeze(0))

    pred = logits.softmax(-1)
    label, confidence = ocr.tokenizer.decode(pred)
    print('Decoded plate: {}'.format(label[0]))
