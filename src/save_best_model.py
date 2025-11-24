import os
import json
import tensorflow as tf
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Modelo")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoints", "cp-41.weights.h5")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "herramientas.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

def save_best():
    print("Cargando labels...")
    with open(LABELS_PATH, 'r') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    print(f"Num classes: {num_classes}")
    
    print("Construyendo modelo...")
    model = build_model(num_classes, 224)
    
    print(f"Cargando pesos de {CHECKPOINT_PATH}...")
    model.load_weights(CHECKPOINT_PATH)
    
    print(f"Guardando modelo completo en {OUTPUT_MODEL_PATH}...")
    model.save(OUTPUT_MODEL_PATH)
    print("¡Listo!")

if __name__ == "__main__":
    save_best()
