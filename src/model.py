import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes, img_size):
    """
    Crea un modelo de clasificación de imágenes basado en MobileNetV2.
    num_classes: número de clases en tu dataset.
    img_size: tamaño de la imagen (img_size x img_size x 3).
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Descongelar las capas superiores para fine-tuning
    base_model.trainable = True
    
    # Congelar todas las capas excepto las últimas 50
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    
    # IMPORTANTE: Usar la función de preprocesamiento de MobileNetV2
    # Esta función espera valores [0, 255] y los escala a [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False) # training=False es importante para BatchNorm

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)  # Aumentar dropout un poco
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model