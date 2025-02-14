import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen y convertirla a escala de grises
imagen = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)

# Función para mostrar varias imágenes con sus títulos
def mostrar_imagenes(imagenes, titulos):
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(imagenes):
        plt.subplot(1, len(imagenes), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(titulos[i], fontsize=10)
        plt.axis('off')
    plt.show()

# a) Reducir la resolución en factores K = 2, 3 y 4
imagenes_reducidas = []
titulos_reducidos = []
for k in [2, 3, 4]:
    nueva_imagen = cv2.resize(imagen, (imagen.shape[1] // k, imagen.shape[0] // k))
    imagenes_reducidas.append(nueva_imagen)
    titulos_reducidos.append(f"Dim: {nueva_imagen.shape[1]}x{nueva_imagen.shape[0]}")

mostrar_imagenes(imagenes_reducidas, titulos_reducidos)

# b) Aumentar la resolución en factores K = 10, 12 y 16
imagenes_aumentadas = []
titulos_aumentados = []
for k in [10, 12, 16]:
    nueva_imagen = cv2.resize(imagen, (imagen.shape[1] * k, imagen.shape[0] * k))
    imagenes_aumentadas.append(nueva_imagen)
    titulos_aumentados.append(f"Dim: {nueva_imagen.shape[1]}x{nueva_imagen.shape[0]}")

mostrar_imagenes(imagenes_aumentadas, titulos_aumentados)

# 2) Rotar la imagen a 90°, 75°, 47° y 135°
angulos = [90, 75, 47, 135]
imagenes_rotadas = []
titulos_rotados = []

for angulo in angulos:
    M = cv2.getRotationMatrix2D((imagen.shape[1] // 2, imagen.shape[0] // 2), angulo, 1)
    nueva_imagen = cv2.warpAffine(imagen, M, (imagen.shape[1], imagen.shape[0]))
    imagenes_rotadas.append(nueva_imagen)
    titulos_rotados.append(f"Rotación: {angulo}°")

mostrar_imagenes(imagenes_rotadas, titulos_rotados)

# d) Calcular el histograma de la imagen (sin usar funciones predefinidas)
histograma = np.zeros(256, dtype=int)

# Recorrer cada píxel de la imagen y calcular el histograma
for fila in range(imagen.shape[0]):
    for columna in range(imagen.shape[1]):
        valor = imagen[fila, columna]
        histograma[valor] += 1

# Mostrar el histograma
plt.figure(figsize=(10, 5))
plt.bar(np.arange(256), histograma, color='gray')
plt.title("Histograma Calculado Manualmente")
plt.xlabel("Niveles de gris")
plt.ylabel("Cantidad de píxeles")
plt.grid(True)
plt.show()