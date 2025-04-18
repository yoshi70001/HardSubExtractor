import cv2
import time

# Cargar modelo FP32
net_fp32 = cv2.dnn.readNet("modelo_fp32.onnx")
net_fp32.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCL)
net_fp32.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Cargar modelo FP16
net_fp16 = cv2.dnn.readNet("modelo_fp16.onnx")
net_fp16.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCL)
net_fp16.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# Imagen de prueba
blob = cv2.dnn.blobFromImage(cv2.imread("imagen.jpg"), scalefactor=1.0, size=(640, 640))

# Medir tiempo FP32
start = time.time()
net_fp32.setInput(blob)
net_fp32.forward()
print("Tiempo FP32:", time.time() - start)

# Medir tiempo FP16
start = time.time()
net_fp16.setInput(blob)
net_fp16.forward()
print("Tiempo FP16:", time.time() - start)
