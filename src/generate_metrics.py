import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")
MODEL_DIR = os.path.join(BASE_DIR, "Modelo")
MODEL_PATH = os.path.join(MODEL_DIR, "herramientas.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    print("===== GENERANDO MÉTRICAS =====")
    
    # 1. Cargar etiquetas
    if not os.path.exists(LABELS_PATH):
        print("Error: No se encontró labels.json")
        return
        
    with open(LABELS_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Clases: {len(class_names)}")

    # 2. Preparar generador de validación (SIN data augmentation)
    # Solo validation_split, sin rotaciones ni zooms para evaluar la realidad
    datagen = ImageDataGenerator(validation_split=0.2)

    print("Cargando dataset de validación...")
    val_ds = datagen.flow_from_directory(
        DATASET_DIR,
        subset="validation",
        seed=42,
        shuffle=False, # Importante para que el orden coincida con las predicciones
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # 3. Construir modelo y cargar pesos
    print("Construyendo modelo...")
    # Definir build_model aquí para evitar problemas de importación
    def build_model(num_classes, img_size):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    try:
        model = build_model(len(class_names), IMG_SIZE)
        print(f"Cargando pesos de {MODEL_PATH}...")
        model.load_weights(MODEL_PATH)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    # 4. Predecir
    print("Realizando predicciones...")
    y_prob = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_ds.classes

    # 5. Generar Reporte
    print("Generando reporte...")
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    report_path = os.path.join(MODEL_DIR, "reporte_clasificacion.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Reporte guardado en: {report_path}")
    print(report)

    # 6. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(MODEL_DIR, "matriz_confusion.csv")
    np.savetxt(cm_path, cm, delimiter=",")
    print(f"Matriz de confusión guardada en: {cm_path}")

if __name__ == "__main__":
    main()
