import numpy as np
from PIL import Image

from rfdetr import RFDETRNano
from paddleocr import PaddleOCR
from modules import ocr_module

plate_detect_model = RFDETRNano(pretrain_weights="weights/best.pth")
plate_detect_model.optimize_for_inference()

ocr = PaddleOCR(lang='en')

img = ocr_module.read_image("auto.jpg")
plate_images = ocr_module.detect_registration_plates(img=img, 
                                                     detector=plate_detect_model)
licence_plate_numbers: list = []
for plate in plate_images:
            reg_text = ocr_module.ocr_registration_plates(img=plate,
                                                          ocr=ocr)
            licence_plate_numbers.append(reg_text)

print(licence_plate_numbers)


