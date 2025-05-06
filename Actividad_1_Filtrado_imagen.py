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

    # Imagen a Blanco y negro
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Filtramos valores de negros, setting pixeles con una intensidad MINUMA de 134 a 255
    thresh_val = 134
    _, mascara_binaria = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY_INV)

    #Aplicamos gausian blur de 31x31 para deshacernos de la presencia de letras pequeñas (Darwinismo)
    blurred = cv.GaussianBlur(mascara_binaria, (31, 31), 0)

    #Reaplicamos filtro
    _, final_mask = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)

    #Slideshow de progreso
    cv.imshow("Original", img)
    cv.waitKey(0)
    cv.imshow("Blanco y Negro", mascara_binaria)
    cv.waitKey(0)
    cv.imshow("Filtro Gaussiano", blurred)
    cv.waitKey(0)
    cv.imshow("Final Mask", final_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Lectua de imagen
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    texts = reader.readtext(final_mask, detail=0)
    detected = " ".join(texts)

    print("Texto encontrado:\n")
    #Final
    letters = [c for c in detected if c.isalpha()]
    if not letters:
        print("No se detecto texto.")
    else:
        print("".join(letters))
        
first_plate = cv.imread("placa_q.jpg")
second_plate = cv.imread("placa_4.jpg")
if first_plate is None:
    raise FileNotFoundError("No se encontró 'first_plate '.")
if second_plate is None:
    raise FileNotFoundError("No se encontró 'second_plate'.")

print("_________________________________________________________________________________")
print("Lectura de Imagen 1 (presiona 0 para continuar):")
cv.waitKey(0)
filter_and_read(first_plate )
print("_________________________________________________________________________________")
print("Lectura de Imagen 2 (presiona 0 para continuar):")
cv.waitKey(0)
filter_and_read(second_plate)
print("_________________________________________________________________________________")