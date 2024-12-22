import tensorflow as tf
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split


def predict_image(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    img = tf.image.resize(img, (28, 28)).numpy()
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    result = {}
    for i in range(10):
        result[str(i)] = float(prediction[0][i])
    return result


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# dataAug = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=5,
#     zoom_range=0.05,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )
# dataAug.fit(x_train)

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(image_mode='L'),
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title="Digit Classifier",
)

interface.launch()
