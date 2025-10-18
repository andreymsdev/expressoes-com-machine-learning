import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ============================
# 1. TREINAMENTO DO MODELO
# ============================

# Caminho para o dataset FER2013
base_path ="datasets/fer2013/train"

# Pré-processamento das imagens
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Criação da rede neural convolucional
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 emoções
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
print("Treinando o modelo...")
model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save("modelo_emocoes.h5")
print("Modelo treinado e salvo como 'modelo_emocoes.h5'.")

# ============================
# 2. CAPTURA COM OPENCV
# ============================

# Lista de emoções (de acordo com as pastas do dataset)
emocoes = sorted(os.listdir(base_path))

# Carrega o modelo treinado
from tensorflow.keras.models import load_model
model = load_model("modelo_emocoes.h5")

# Inicia a webcam
cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensiona para 48x48 (tamanho usado no treinamento)
    face = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0

    # Faz a predição
    pred = model.predict(face)
    emocao = emocoes[np.argmax(pred)]

    # Mostra o resultado na tela
    cv2.putText(frame, f"Expressão: {emocao}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Reconhecimento Facial", frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
