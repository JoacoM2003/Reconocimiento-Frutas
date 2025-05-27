import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocesamiento
train_path = 'frutas/Training'
test_path = 'frutas/Test'

img_size = (100, 100)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255)  
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    test_path, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# Modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(train_data, epochs=10, validation_data=test_data)

# Guardar el modelo entrenado
model.save('modelo_frutas_cnn.h5')
