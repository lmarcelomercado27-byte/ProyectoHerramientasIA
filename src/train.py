print(">>> train.py se está ejecutando <<<")

import os
import json
import traceback

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")
MODEL_DIR = os.path.join(BASE_DIR, "Modelo")

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10


def main():
    print("===== INICIANDO ENTRENAMIENTO =====")
    print("Python:", os.sys.executable)
    print("TensorFlow:", tf.__version__)
    print("BASE_DIR:", BASE_DIR)
    print("DATASET_DIR:", DATASET_DIR)

    # Comprobar que existe el dataset
    if not os.path.isdir(DATASET_DIR):
        print("ERROR: La carpeta 'DataSet' NO existe.")
        return

    subdirs = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print("Subcarpetas encontradas en dataset:", subdirs)

    if not subdirs:
        print("ERROR: 'DataSet' no tiene subcarpetas (clases).")
        return

    # ================= GENERADORES =================
    # Entrenamiento: con data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    # Validación: sin augmentation
    val_datagen = ImageDataGenerator(
        validation_split=0.2
    )

    print("Creando generadores de datos...")

    train_ds = train_datagen.flow_from_directory(
        DATASET_DIR,
        subset="training",
        seed=42,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Para métricas correctas usamos shuffle=False en validación
    val_ds = val_datagen.flow_from_directory(
        DATASET_DIR,
        subset="validation",
        seed=42,
        shuffle=False,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    num_classes = train_ds.num_classes
    class_names = list(train_ds.class_indices.keys())

    print("Número de clases:", num_classes)
    print("Clases:", class_names)

    print("Construyendo modelo...")
    model = build_model(num_classes, IMG_SIZE)

    # ================= CHECKPOINTS Y CALLBACKS =================
    checkpoint_dir = os.path.join(MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:02d}.weights.h5")

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    # ==========================================================

    print("Comenzando entrenamiento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stopping, reduce_lr]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "herramientas.h5")
    labels_path = os.path.join(MODEL_DIR, "labels.json")

    model.save(model_path)

    with open(labels_path, "w") as f:
        json.dump(class_names, f)

    print("===== ENTRENAMIENTO COMPLETADO =====")
    print("Modelo guardado en:", model_path)
    print("Labels guardados en:", labels_path)

    # ========== MÉTRICAS Y GRÁFICOS ==========

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    # Gráfico de accuracy
    if acc and val_acc:
        plt.figure(figsize=(8, 5))
        plt.plot(acc, label="accuracy")
        plt.plot(val_acc, label="val_accuracy")
        plt.title("Curva de Precisión")
        plt.xlabel("Época")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "curva_accuracy.png"))
        plt.close()

    # Gráfico de loss
    if loss and val_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val_loss")
        plt.title("Curva de Pérdida")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "curva_loss.png"))
        plt.close()

    # ====== MATRIZ DE CONFUSIÓN Y REPORTE ======
    print("Calculando métricas de validación (matriz de confusión y reporte)...")

    val_ds.reset()
    y_true = val_ds.classes  # etiquetas verdaderas
    y_prob = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(MODEL_DIR, "matriz_confusion.csv"), cm, delimiter=",")

    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(MODEL_DIR, "reporte_clasificacion.txt"), "w") as f:
        f.write(report)

    print("Matriz de confusión y reporte de clasificación guardados en la carpeta Modelo.")
    # ==========================================


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR durante la ejecución de train.py")
        traceback.print_exc()
