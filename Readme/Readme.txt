# Proyecto de Clasificación de Herramientas con CNN

Este proyecto detecta y clasifica herramientas utilizando Deep Learning con TensorFlow y la cámara del computador.

## 📂 Estructura

- `dataset/`: Contiene 40 clases con imágenes.
- `src/train.py`: Entrenamiento del modelo.
- `src/predict.py`: Predicción desde archivo.
- `src/webcam.py`: Detección en tiempo real con cámara.
- `modelo/`: Modelo entrenado (.h5) y labels.

## ▶ Entrenar:

```bash
python src/train.py