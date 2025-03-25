from paddleocr import PaddleOCR
import cv2
ocr = PaddleOCR(lang='es',use_angle_cls=False,gpu_mem=5*1024,show_log = False) 
import numpy as np
def predict( img):
    result = ocr.ocr(img,det=True,rec=False,cls=False)
    return result

def predict_and_detect(img, rectangle_thickness=2, text_thickness=1):
    boxes = predict( img)
    
    if boxes[0]:
        polygons = [np.array(polygon, dtype=np.int32) for polygon in boxes[0]]
        for polygon in polygons:
            cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                

input_filename = "videos/AormeSubs_Araiya_san!_Ore_to_Aitsu_ga_Onnayu_de!_01_SIN_CENSURA.mp4"
cap = cv2.VideoCapture(input_filename,cv2.CAP_FFMPEG)
heigth = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
counterFrame = 1
while True:
    success, img = cap.read()
    if not success:
        break
    if counterFrame%5==0:
        img = cv2.resize(img,(int(width*(360/heigth)),360),interpolation=cv2.INTER_AREA)
        predict_and_detect(img)
        # writer.write(result_img)
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
    counterFrame+=1
cap.release()