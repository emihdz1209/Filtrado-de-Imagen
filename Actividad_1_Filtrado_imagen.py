"""
Módulo para detección y lectura OCR de placas usando OpenCV y EasyOCR.

Contiene la función `filter_and_read` que aplica filtros y realiza OCR
sobre una imagen para extraer únicamente caracteres alfabéticos.
"""

import cv2 as cv
import numpy as np
import easyocr
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader"
)

def filter_and_read(img):
    """
    Aplica filtros para resaltado de texto en una imagen y ejecuta OCR.

    Parámetros usados:
        -img: Imagen de entrada.

    Proceso:
        1. Conversión a escala de grises.
        2. Resaltar regiones oscuras (texto).
        3. Desenfoque Gaussiano para eliminar texto diminuto.
        4. Segundo umbral para refinar la nitidez del texto central.
        5. Muestra cada etapa como ventana interactiva.
        6. Ejecuta EasyOCR sobre la máscara final e imprime letras en consola.
    """
    # 1. Escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Filtro inverso para extraer texto oscuro
    thresh_val = 134
    _, mascara_binaria = cv.threshold(
        gray,
        thresh_val,
        255,
        cv.THRESH_BINARY_INV
    )

    # 3. Desenfoque para eliminar texto muy pequeño
    blurred = cv.GaussianBlur(mascara_binaria, (31, 31), 0)

    # 4. Segundo umbral 
    _, final_mask = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)

    # 5. Mostrar cada paso 
    cv.imshow("Original", img)
    cv.waitKey(0)
    cv.imshow("Blanco y Negro", mascara_binaria)
    cv.waitKey(0)
    cv.imshow("Filtro Gaussiano", blurred)
    cv.waitKey(0)
    cv.imshow("Final Mask", final_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 6. Lectura OCR y filtrado de caracteres alfabéticos
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    texts = reader.readtext(final_mask, detail=0)
    detected = " ".join(texts)

    print("Texto encontrado:\n")
    letters = [c for c in detected if c.isalpha()]
    if not letters:
        print("No se detectó texto.")
    else:
        print("".join(letters))
        
first_plate = cv.imread("placa_q.jpg")
second_plate = cv.imread("placa_4.jpg")
if first_plate is None:
    raise FileNotFoundError("No se encontró 'first_plate '.")
if second_plate is None:
    raise FileNotFoundError("No se encontró 'second_plate'.")

line="_______________________________________________________________"

print(line)
print("Lectura de Imagen 1 (presiona 0 para continuar):")
cv.waitKey(0)
filter_and_read(first_plate )
print(line)
print("Lectura de Imagen 2 (presiona 0 para continuar):")
cv.waitKey(0)
filter_and_read(second_plate)
print(line)