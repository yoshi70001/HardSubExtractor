import cv2
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR

# Inicializa PaddleOCR globalmente
ocr = PaddleOCR(lang='es', use_angle_cls=False, show_log=False, use_gpu=True)

def initialize_session(model_path="models/model.onnx"):
    """
    Inicializa la sesión ONNX utilizando GPU si está disponible, y retorna
    la sesión junto con los nombres de entrada y salida del modelo.
    """
    options = ort.SessionOptions()
    available_providers = ort.get_available_providers()
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else available_providers
    session = ort.InferenceSession(model_path, options, providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess_frame(frame, size=(224, 224)):
    """
    Redimensiona el frame y obtiene sus versiones en escala de grises y RGB.
    """
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return gray, rgb

def run_batch_inference(session, input_name, output_name, rgb_frames_batch):
    """
    Realiza la inferencia en batch sobre una lista de frames RGB preprocesados.
    Normaliza, reorganiza el tensor a formato CHW y ejecuta el modelo.
    Luego, aplica postprocesamiento (umbral, erosión y bitwise_not) a cada resultado.
    """
    # Convertir cada frame a formato CHW, normalizar y asegurar el tipo float32
    chw_batch = [
        np.transpose(frame / 255.0, (2, 0, 1)).astype(np.float32)
        for frame in rgb_frames_batch
    ]
    batch_input = np.stack(chw_batch, axis=0)  # Forma: (batch, C, H, W)
    
    outputs = session.run([output_name], {input_name: batch_input})[0]
    processed_outputs = []
    for output in outputs:
        # Reorganizar de CHW a HWC y aplicar umbral y postprocesamiento
        out = np.transpose(output, (1, 2, 0))
        out = (out > 0.5).astype(np.uint8) * 255
        out = cv2.erode(out, np.ones((3, 6), np.uint8), iterations=2)
        out = cv2.bitwise_not(out)
        processed_outputs.append(out)
    return processed_outputs

def process_video(video_path, process_every_n_frame=2, batch_size=100, similar_pixel_threshold=55, similar_image_threshold=300):
    """
    Procesa el video, aplicando filtrado para detectar frames con cambios significativos,
    ejecuta inferencia en batch y, si se cumple la condición de OCR, extrae texto del frame original.
    """
    session, input_name, output_name = initialize_session()
    video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    prev_gray = None
    frame_counter = 0
    # batch_frames almacenará tuplas: (frame_original, rgb_frame)
    batch_frames = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_counter % process_every_n_frame == 0:
                # Preprocesamiento en paralelo
                future = executor.submit(preprocess_frame, frame)
                gray, rgb_frame = future.result()
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, absdiff = cv2.threshold(diff, similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                    # Se acumula el frame si el cambio es significativo
                    if np.count_nonzero(absdiff) >= similar_image_threshold:
                        batch_frames.append((frame, rgb_frame))
                        # Cuando el batch alcanza el tamaño, se procesa la inferencia
                        if len(batch_frames) >= batch_size:
                            rgb_batch = [item[1] for item in batch_frames]
                            outputs = run_batch_inference(session, input_name, output_name, rgb_batch)
                            for (orig_frame, _), output in zip(batch_frames, outputs):
                                summary = np.sum(output)
                                # print(f"Output summary: {summary}")
                                # Condición para ejecutar OCR (se puede ajustar el umbral)
                                if summary < (224 * 224 * 255) - 35700:
                                    texts = []
                                    result = ocr.ocr(orig_frame, det=True, rec=True, cls=False)
                                    if result and result[0]:
                                        for line in result[0]:
                                            if line[1][1] * 100 > 70:
                                                texts.append(line[1][0])
                                    text_content = ' '.join(texts)
                                    if texts:
                                        print(text_content)
                            batch_frames = []  # Reinicia el batch
                prev_gray = gray
            frame_counter += 1
        
        # Procesa cualquier batch restante
        if batch_frames:
            rgb_batch = [item[1] for item in batch_frames]
            outputs = run_batch_inference(session, input_name, output_name, rgb_batch)
            for (orig_frame, _), output in zip(batch_frames, outputs):
                summary = np.sum(output)
                # print(f"Output summary: {summary}")
                if summary < (224 * 224 * 255) - 35700:
                    texts = []
                    result = ocr.ocr(orig_frame, det=True, rec=True, cls=False)
                    if result and result[0]:
                        for line in result[0]:
                            if line[1][1] * 100 > 70:
                                texts.append(line[1][0])
                    text_content = ' '.join(texts)
                    if texts:
                        print(text_content)
    video.release()

if __name__ == '__main__':
    video_path = '[AU-TnF]Lost Universe 18 [602C8C6F] By Julian12100.avi'
    process_video(video_path)
