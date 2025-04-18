import onnx
from onnxconverter_common import float16

def convert_onnx_to_fp16(onnx_model_path: str, fp16_model_path: str):
    # Cargar el modelo ONNX original
    model = onnx.load(onnx_model_path)
    
    # Convertir a FP16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Guardar el modelo convertido
    onnx.save(model_fp16, fp16_model_path)
    print(f"Modelo convertido a FP16 guardado en: {fp16_model_path}")

# Rutas de entrada y salida
onnx_model_path = "models/model.onnx"
fp16_model_path = "models/model_fp16.onnx"

# Llamar a la función de conversión
convert_onnx_to_fp16(onnx_model_path, fp16_model_path)
