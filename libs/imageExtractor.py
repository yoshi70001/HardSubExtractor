import cv2
import numpy as np
import onnxruntime as ort
from datetime import timedelta

# Configuración de la sesión ONNX
options = ort.SessionOptions()
providers = ort.get_available_providers()
model_path = "models/model.onnx"
session = ort.InferenceSession(model_path, options, providers)
frameBuffer = []
# Obtener los nombres de las entradas y salidas del modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"

def scale_contours(contours, original_width, original_height):
    # Escalar los contornos a la resolución original
    scale_x = original_width / 224
    scale_y = original_height / 224
    scaled_contours = []
    for contour in contours:
        contour = contour.astype(np.float32)
        contour[:, :, 0] *= scale_x  # Escalar coordenadas x
        contour[:, :, 1] *= scale_y  # Escalar coordenadas y
        scaled_contours.append(contour.astype(np.int32))
    return scaled_contours

def compare_contours(contours1, contours2, threshold=0.7):
    # Si no hay contornos en una de las imágenes, son diferentes
    if len(contours1) == 0 or len(contours2) == 0:
        return False
    
    # Comparar cada contorno de la imagen actual con los de la imagen anterior
    for contour1 in contours1:
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        area1 = w1 * h1
        
        for contour2 in contours2:
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            area2 = w2 * h2
            
            # Calcular la intersección
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                continue  # No hay intersección
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            union_area = area1 + area2 - intersection_area
            
            iou = intersection_area / union_area
            
            # Si la IoU es mayor que el umbral, los contornos son similares
            if iou >= threshold:
                return True
    
    # Si no se encontraron contornos similares, las imágenes son diferentes
    return False

def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0
        chw_frame = np.transpose(normalized_frame, (2, 0, 1))
        processed_frames.append(chw_frame)
    return np.array(processed_frames).astype(np.float32)

def draw_contours_on_image(image, contours):
    # Crear una copia de la imagen para no modificar la original
    image_with_rectangles = image.copy()
    
    # Dibujar un rectángulo alrededor de cada contorno
    for contour in contours:
        # Obtener las coordenadas del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        
        # Dibujar el rectángulo en la imagen
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Color verde, grosor 2
    
    return image_with_rectangles

# Postprocesar la salida
def postprocess_output(output):
    # output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))
    return (output > 0.5).astype(np.uint8) * 255

def proccessFrames(frames,batch_size,counters,fps):
    original_width, original_height = frames[0].shape[1], frames[0].shape[0]
    input_data = preprocess_frames(frames)
    output = session.run([output_name], {input_name: input_data})[0]
    framePositions = []
    for i in range(batch_size):
        output_mask = postprocess_output(output[i])
        inferedFrameEroded = cv2.erode(output_mask,np.ones((3,6)),iterations=2)
        inferedFrameEroded= np.uint8(inferedFrameEroded)
        inferedFrameEroded = cv2.bitwise_not(inferedFrameEroded) # type: ignore
        contours, hierarchy=cv2.findContours(inferedFrameEroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        containText = False
        cv2.imshow('original',frames[i])
        cv2.imshow('mascara',inferedFrameEroded)
        for contour in contours:
            # x,y,w,h = cv2.boundingRect(contour)
            # if h>7 and y >150:
            area = cv2.contourArea(contour)
            if 240< area<112*224:
                containText = True
                break
        if containText :
            # framePositions.append({'counter':counters[i],"frame":frames[i],"fps":fps,"time":format_timedelta(timedelta(seconds=(counters[i]/fps)))})
            cv2.imwrite(f'frames/{format_timedelta(timedelta(seconds=(counters[i]/fps)))}.jpeg',frames[i])
        cv2.waitKey(1)
    
    return framePositions
