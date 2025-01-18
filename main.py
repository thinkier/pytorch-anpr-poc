from ultralytics import YOLO

lpr = YOLO("yolov8x-supervision-license-plate-recognition best.pt")

import torch

ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
from strhub.data.module import SceneTextDataModule

img_transform = SceneTextDataModule.get_transform(ocr.hparams.img_size)

import io
from pathlib import Path
from PIL import Image

directory = Path("in")
with open("out/results.csv", "w") as res_csv:
    res_csv.write("image,plate,plate_confidence,ocr_confidence\r\n")
    for item in directory.iterdir():
        with open(item, "rb") as f:
            image_buffer = io.BytesIO(f.read())
        image = Image.open(image_buffer).convert("RGB")

        plates = lpr.predict(image, conf=0.6)

        for aabb, conf in zip(plates[0].boxes.xyxy, plates[0].boxes.conf):
            x1, y1, x2, y2 = aabb.tolist()

            image_plate = image.crop((x1, y1, x2, y2))
            logits = ocr(img_transform(image_plate).unsqueeze(0))

            pred = logits.softmax(-1)
            label, confidence = ocr.tokenizer.decode(pred)
            res_csv.write('{},{},{},{}\r\n'.format(item, label[0], conf, confidence[0][0]))
