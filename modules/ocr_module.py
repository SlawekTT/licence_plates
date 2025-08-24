import numpy as np
from PIL import Image

from rfdetr import RFDETRNano
from paddleocr import PaddleOCR

def read_image(filename: str)->Image.Image:
    return Image.open(fp=filename)

def detect_vehicles(img: Image.Image, 
                    detector: RFDETRNano)-> list:
    classes_of_interest_ids: list[int] = list(range(2, 9))
    car_images: list = []
    result = detector.predict(img)
    for box, class_id in zip(result.xyxy, result.class_id):
        if class_id in classes_of_interest_ids:
            car_box: list = list(map(int, box))
            car_img: Image.Image = Image.fromarray(np.array(img)[car_box[1]: car_box[3], 
                                                                 car_box[0]: car_box[2], :])
            car_images.append(car_img)
    return car_images

def detect_registration_plates(img: Image.Image, 
                               detector: RFDETRNano)-> list:
    result = detector.predict(img)
    plate_images: list = []
    for box in result.xyxy:
        plate_box: list = list(map(int, box))
        plate_img: Image.Image = Image.fromarray(np.array(img)[plate_box[1]: plate_box[3], 
                                                               plate_box[0]: plate_box[2], :])
        plate_images.append(plate_img)
    return plate_images

def ocr_registration_plates(img: Image.Image, 
                            ocr:PaddleOCR)->str:
    result = ocr.predict(np.array(img))
    return result[0]['rec_texts']