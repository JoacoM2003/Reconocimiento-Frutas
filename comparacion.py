import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Parámetros comunes
train_path = 'frutas/Training'
test_path = 'frutas/Test'
img_size = (100, 100)
batch_size = 32
epochs = 10

# Generadores
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True
)
test_data = test_gen.flow_from_directory(
    test_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

num_classes = train_data.num_classes

# --- Modelo CNN ---
def crear_modelo_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Modelo MLP ---
def crear_modelo_mlp():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar y guardar el historial de ambos modelos
model_cnn = crear_modelo_cnn()
history_cnn = model_cnn.fit(train_data, epochs=epochs, validation_data=test_data)

model_mlp = crear_modelo_mlp()
history_mlp = model_mlp.fit(train_data, epochs=epochs, validation_data=test_data)

# Evaluar ambos modelos
loss_cnn, acc_cnn = model_cnn.evaluate(test_data)
loss_mlp, acc_mlp = model_mlp.evaluate(test_data)

print(f"CNN - Loss: {loss_cnn:.4f}, Accuracy: {acc_cnn:.4f}")
print(f"MLP - Loss: {loss_mlp:.4f}, Accuracy: {acc_mlp:.4f}")

# Función para mostrar matriz de confusión y reporte
def evaluar_modelo(model, test_data, nombre_modelo):
    y_pred_prob = model.predict(test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_data.classes
    class_names = list(test_data.class_indices.keys())
    
    print(f"\nReporte clasificación para {nombre_modelo}:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.show()

# Mostrar resultados detallados
evaluar_modelo(model_cnn, test_data, "CNN")
evaluar_modelo(model_mlp, test_data, "MLP")

# Graficar curvas de aprendizaje
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history_cnn.history['accuracy'], label='CNN Train')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val')
plt.plot(history_mlp.history['accuracy'], label='MLP Train')
plt.plot(history_mlp.history['val_accuracy'], label='MLP Val')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history_cnn.history['loss'], label='CNN Train')
plt.plot(history_cnn.history['val_loss'], label='CNN Val')
plt.plot(history_mlp.history['loss'], label='MLP Train')
plt.plot(history_mlp.history['val_loss'], label='MLP Val')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
