import cv2
import numpy as np
import datetime
import onnxruntime as ort
# from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor
print(datetime.datetime.now())
# ocr = PaddleOCR(lang='es',use_angle_cls=False,show_log = False,use_gpu=True)

executor = ThreadPoolExecutor(max_workers=2)
def ocrp(frame):
    # result = ocr.ocr(frame,det=True,rec=True,cls=False)
    # if result and result[0]:
    #     for line in result[0]:
    #         if line[1][1]*100 > 70:
    #             texts.append(line[1][0])
    # text_content = ' '.join(texts)
    # if len(texts)>0:
    #     print(text_content)
    return 0
def cropImage(frameResized):
    # Crear una máscara negra en los 2/3 superiores
    height = frameResized.shape[0]
    black_region = int((4/5) * height)
    # frameResized[:black_region, :] = 0  # Asignar negro a los 2/3 superiores
    return frameResized

def resize480(frame):
    # Obtener dimensiones originales
    (h, w) = frame.shape[:2]

    # Nueva altura fija
    new_height = 480

    # Calcular nuevo ancho manteniendo la proporción
    aspect_ratio = w / h
    new_width = int(new_height * aspect_ratio)

    # Redimensionar la imagen
    frameResized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return frameResized


# Configuración de la sesión ONNX
options = ort.SessionOptions()
providers = ort.get_available_providers()
model_path = "models/model.onnx"
session = ort.InferenceSession(model_path, options, providers)
frameBuffer = []
# Obtener los nombres de las entradas y salidas del modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

video = cv2.VideoCapture("videos/AormeSubs_Araiya_san!_Ore_to_Aitsu_ga_Onnayu_de!_01_SIN_CENSURA.mp4",cv2.CAP_FFMPEG)
similar_pixel_threshold = 80
similar_image_threshold = 300
prev_grey = None
frameCounter = 0
while video.isOpened():
    ret,frame = video.read()
    if not ret:
        break
    if(ret and frameCounter%2 == 0):
        frameResized= cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR)
        frameResized = cropImage(frameResized)
        frame = resize480(frame)
        frame = cropImage(frame)
        grey = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
        if prev_grey is not None:
            _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey), similar_pixel_threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('absdiff',absdiff)
            if not np.count_nonzero(absdiff) < similar_image_threshold:
                # print('cambio')
                rgb_frame = cv2.cvtColor(frameResized, cv2.COLOR_BGR2RGB)
                normalized_frame = rgb_frame / 255.0 # type: ignore
                chw_frame = np.transpose(normalized_frame, (2, 0, 1))
                output = session.run([output_name], {input_name: [chw_frame]})[0][0]
                output = np.transpose(output, (1, 2, 0))
                output = (output > 0.5).astype(np.uint8) * 255
                output = cv2.erode(output,np.ones((3,6)),iterations=2)
                output = cv2.bitwise_not(output)
                summary = np.sum(output)
                cv2.imshow('output',output)
                # print( summary<(224*224*255)-35700)
                texts =[]
                if summary<(224*224*255)-35700:
                    future = executor.submit(ocrp, frame)
                    # gray, rgb_frame = future.result()
                    # result = ocr.ocr(frame,det=True,rec=True,cls=False)
                    # if result and result[0]:
                    #     for line in result[0]:
                    #         if line[1][1]*100 > 70:
                    #             texts.append(line[1][0])
                    # text_content = ' '.join(texts)
                    # if len(texts)>0:
                    #     print(text_content)
                    # cv2.imwrite(f'frames/{frameCounter}.jpeg',frame)
        cv2.imshow('original',frameResized)
        cv2.waitKey(1)
        prev_grey = grey
    frameCounter+=1
print(datetime.datetime.now())