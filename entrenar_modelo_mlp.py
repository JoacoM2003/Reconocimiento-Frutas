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

# Modelo MLP
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),        # Aplana la imagen 100x100x3 = 30000
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')  # Número de clases detectado automáticamente
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(train_data, epochs=10, validation_data=test_data)

# Guardar el modelo
model.save('modelo_frutas_mlp.h5')