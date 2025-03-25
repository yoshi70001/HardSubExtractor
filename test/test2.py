import cv2
import numpy as np
import onnxruntime as ort
from datetime import timedelta

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='es',use_angle_cls=False,gpu_mem=5*1024,show_log = False) 
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
    for index in range(batch_size):
        frame = frames[index]
        frame = cv2.resize(frame,(int((480/original_height)*original_width),480),interpolation=cv2.INTER_NEAREST)
        boxes = ocr.ocr(frame,det=True,rec=False,cls=False)
        # if boxes[0]:
        #     polygons = [np.array(polygon, dtype=np.int32) for polygon in boxes[0]]
        #     for polygon in polygons:
        #         cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        if(boxes[0]):
            frameTime = frameTimes[index]
            # print(boxes[0])
            current_time = timedelta(seconds=(frameTime / fps))
            print(current_time)
        cv2.imshow('original', frame)
        cv2.waitKey(1)

    

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
batch_size = 30
all_subtitles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if counterFrame%4==0:
        frames.append(frame)
        frameTimes.append(counterFrame)
        if len(frames) == batch_size:
            subtitles = process_frames(frames, batch_size, fps,frameTimes)
            frames = []
            frameTimes = []
    counterFrame+=1
cap.release()
cv2.destroyAllWindows()

print(f"Subtítulos guardados en {output_srt_file}")
