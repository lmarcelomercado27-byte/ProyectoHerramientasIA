"""
Servidor Flask para el Clasificador de Herramientas
Proporciona endpoints para la predicción de imágenes usando el modelo entrenado
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Modelo")
MODEL_PATH = os.path.join(MODEL_DIR, "herramientas.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

import sys
sys.path.insert(0, BASE_DIR)
from src.model import build_model

IMG_SIZE = 224

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializar Flask con la carpeta de templates correcta
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# Cargar labels primero para obtener num_classes
try:
    with open(LABELS_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Labels cargados: {len(class_names)} clases")
except Exception as e:
    print(f"✗ Error al cargar labels: {e}")
    class_names = []

# Cargar modelo
print("Cargando modelo...")
model = None
if class_names:
    try:
        # Intentar cargar el modelo completo primero
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✓ Modelo .h5 cargado directamente")
        except:
            # Si falla, reconstruir y cargar pesos
            print("  Reconstruyendo arquitectura y cargando pesos...")
            model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
            model.load_weights(MODEL_PATH)
            print("✓ Modelo reconstruido y pesos cargados")
        
        # Recompilar el modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("✓ Modelo compilado correctamente")
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        import traceback
        traceback.print_exc()
        model = None


def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(img_path):
    """
    Preprocesa la imagen para el modelo
    Args:
        img_path: ruta a la imagen
    Returns:
        array numpy preprocesado
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predicción de imágenes
    Recibe una imagen y devuelve la clasificación
    """
    # Verificar modelo cargado
    if model is None:
        return jsonify({'error': 'El modelo no está disponible'}), 500
    
    # Verificar que se envió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    
    # Verificar que el archivo tiene nombre
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # Verificar extensión
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido. Use: png, jpg, jpeg, gif, bmp, webp'}), 400
    
    try:
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocesar imagen
        img_array = prepare_image(filepath)
        
        # Realizar predicción
        predictions = model.predict(img_array, verbose=0)
        predictions = predictions[0]  # Obtener primera (y única) predicción
        
        # Obtener clase predicha
        pred_idx = np.argmax(predictions)
        pred_class = class_names[pred_idx]
        pred_prob = float(predictions[pred_idx])
        
        # Obtener top 5
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5 = [
            {
                'class': class_names[idx],
                'prob': float(predictions[idx])
            }
            for idx in top5_indices
        ]
        
        # Limpiar archivo temporal (opcional)
        try:
            os.remove(filepath)
        except:
            pass
        
        # Devolver resultado
        return jsonify({
            'pred_class': pred_class,
            'pred_prob': pred_prob,
            'top5': top5
        })
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500


@app.route('/health')
def health():
    """Endpoint para verificar el estado del servidor"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'classes': class_names
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Servidor Flask - Clasificador de Herramientas IA")
    print("="*60)
    print(f"📁 Modelo: {MODEL_PATH}")
    print(f"🏷️  Clases: {len(class_names)}")
    print(f"📤 Carpeta uploads: {UPLOAD_FOLDER}")
    print("="*60)
    print("\n🌐 Iniciando servidor en http://127.0.0.1:5001")
    print("   Presiona Ctrl+C para detener\n")
    
    app.run(debug=True, host='127.0.0.1', port=5001)
