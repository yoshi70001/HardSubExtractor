import cv2
import numpy as np
import onnxruntime as ort
from datetime import timedelta

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='es',use_angle_cls=False,gpu_mem=12000,show_log = False) 
# Configuración de la sesión ONNX para segmentación
segmentation_model_path = "models/model.onnx"
# Configuración de la sesión ONNX
options = ort.SessionOptions()
providers = ort.get_available_providers()
segmentation_session = ort.InferenceSession(segmentation_model_path, options, providers)
segmentation_input_name = segmentation_session.get_inputs()[0].name
segmentation_output_name = segmentation_session.get_outputs()[0].name

prev_mask = None  # Almacenar la máscara del frame anterior
change_threshold = 300  # Umbral de cambio (ajustar según necesidad)

def scale_contours(contours, original_width, original_height):
    scale_x = original_width / 224
    scale_y = original_height / 224
    scaled_contours = []
    for contour in contours:
        contour = contour.astype(np.float32)
        contour[:, :, 0] *= scale_x
        contour[:, :, 1] *= scale_y
        scaled_contours.append(contour.astype(np.int32))
    return scaled_contours

def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0
        chw_frame = np.transpose(normalized_frame, (2, 0, 1))
        processed_frames.append(chw_frame)
    return np.array(processed_frames).astype(np.float32)

def postprocess_output(output):
    output = np.transpose(output, (1, 2, 0))
    return (output > 0.5).astype(np.uint8) * 255

def recognize_text(region):
    result = ocr.ocr(region,det=True,rec=True,cls=False)
    return result

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def process_frames(frames, batch_size, fps,frameTimes):
    original_width, original_height = frames[0].shape[1], frames[0].shape[0]
    input_data = preprocess_frames(frames)
    output = segmentation_session.run([segmentation_output_name], {segmentation_input_name: input_data})[0]
    subtitles = []
    active_phrases = {}
    inactive_phrases ={}
    buffer_time = timedelta(seconds=0.5)  # Ajusta el buffer según necesites (medio segundo en este caso)
    for i in range(batch_size):
        frameTime = frameTimes[i]
        output_mask = postprocess_output(output[i])
        inferedFrameEroded = cv2.erode(output_mask, np.ones((3, 6)), iterations=2)
        inferedFrameEroded = np.uint8(inferedFrameEroded)
        inferedFrameEroded = cv2.bitwise_not(inferedFrameEroded)
        contours, hierarchy = cv2.findContours(inferedFrameEroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        global prev_mask
        global change_threshold
        # Detección de texto
        contains_text = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 7 and y > 150:
                area = cv2.contourArea(contour)
                if 240 < area < 112 * 224:
                    contains_text = True
                    break
        
        # Detectar cambio significativo con el frame anterior
        if prev_mask is not None:
            diff = cv2.absdiff(inferedFrameEroded, prev_mask)
            diff_pixels = cv2.countNonZero(diff)
            has_changed = diff_pixels > change_threshold
        else:
            has_changed = False  # Primer frame del batch
        
        # Ejecutar OCR solo si hay cambio y texto
        if has_changed and contains_text:
            result = recognize_text(frames[i])
            texts = [line[1][0] for line in result[0]] if result and result[0]  else []
            recognized_text = ' '.join(texts)
            
            if recognized_text:
                current_time = timedelta(seconds=(frameTime / fps))
                if recognized_text not in active_phrases:
                    active_phrases[recognized_text] = (current_time, current_time)
                else:
                    active_phrases[recognized_text] = (active_phrases[recognized_text][0], current_time)

        # Actualizar la máscara anterior
        prev_mask = inferedFrameEroded.copy()
        
        # Actualizar tiempos de frases activas
        current_time = timedelta(seconds=(frameTime / fps))
        for text in list(active_phrases.keys()):
            start, end = active_phrases[text]
            
            if contains_text and text == recognized_text:
                # Mantener la frase activa
                active_phrases[text] = (start, current_time + buffer_time)
            else:
                # Solo cerrar la frase si ha pasado el buffer
                if current_time - end >= buffer_time:
                    subtitles.append((start, end, text))
                    inactive_phrases[text] = active_phrases[text]
                    del active_phrases[text]

        cv2.imshow('original', frames[i])
        cv2.imshow('mascara', inferedFrameEroded)
        cv2.waitKey(1)

    # Añadir las frases activas restantes
    for text, (start, end) in active_phrases.items():
        subtitles.append((start, end, text))
        print(start, end, text)
    return subtitles

def save_to_srt(subtitles, output_file):
    with open(output_file, 'w') as f:
        for idx, (start_time, end_time, text) in enumerate(subtitles, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timedelta(start_time)} --> {format_timedelta(end_time)}\n")
            f.write(f"{text}\n\n")

# Ejemplo de uso
video_path = "videos/11frieren_EDIT.mp4"
output_srt_file = "output.srt"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
frameTimes =[]
counterFrame = 1
batch_size = 60
all_subtitles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frameTimes.append(counterFrame)
    if len(frames) == batch_size:
        subtitles = process_frames(frames, batch_size, fps,frameTimes)
        all_subtitles.extend(subtitles)
        frames = [frame]
        frameTimes = [counterFrame]
    counterFrame+=1
cap.release()
cv2.destroyAllWindows()

# Guardar todos los subtítulos en un archivo SRT
save_to_srt(all_subtitles, output_srt_file)
print(f"Subtítulos guardados en {output_srt_file}")
