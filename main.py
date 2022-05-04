import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
from skimage.metrics import structural_similarity


def main():
    cargarMascaras(args.train_path)
    detector(args.test_path)  # hacer comparacion, calculo regiones, etc


def cargarMascaras(path):
    try:
        os.makedirs('mascaras')
    except FileExistsError:
        print("Ya existe el directorio")
        # pass

    prohibicion = ["00", "01", "02", "03", "04", "05", "07", "08", "09", "10", "15", "16"]
    resultado = HSV(media(path, prohibicion), 1)
    cv2.imwrite("mascaras/prohibicion.jpg", resultado)

    peligro = ["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    resultado = HSV(media(path, peligro), 1)
    cv2.imwrite("mascaras/peligro.jpg", resultado)

    stop = ["14"]
    resultado = HSV(media(path, stop), 1)
    cv2.imwrite("mascaras/stop.jpg", resultado)

    dirProhibida = ["17"]
    resultado = HSV(media(path, dirProhibida), 1)
    cv2.imwrite("mascaras/dirProhibida.jpg", resultado)

    cedaPaso = ["13"]
    resultado = HSV(media(path, cedaPaso), 1)
    cv2.imwrite("mascaras/cedaPaso.jpg", resultado)

    dirObligatoria = ["38"]
    resultado = HSV(media(path, dirObligatoria), 2)
    cv2.imwrite("mascaras/dirObligatoria.jpg", resultado)


def media(path, nombres):
    suma = cv2.imread(path + "/" + nombres[0] + "/00000.jpg", 1)
    k = 0
    for i in nombres:
        direccion = path + "/" + i
        imagenes = os.listdir(direccion)
        for j in imagenes:
            ## https://stackoverflow.com/questions/57723968/blending-multiple-images-with-opencv ##
            img = cv2.imread(direccion + "/" + j, 1)  # imagen en BGR
            img = cv2.resize(img, (suma.shape[1], suma.shape[0]), interpolation=cv2.INTER_CUBIC)
            k = k + 1
            sc_w = 1 / (k + 1)
            fr_w = 1 - sc_w
            suma = cv2.addWeighted(suma, fr_w, img, sc_w, 0)
            ## - ##

    return suma


def HSV(img, mode):
    img = cv2.resize(img, (25, 25))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if mode == 2:
        ## https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv ##
        lowBlue = np.array([90, 70, 50], np.uint8)
        highBlue = np.array([128, 255, 255], np.uint8)
        maskBlue = cv2.inRange(hsv, lowBlue, highBlue)
        ## - ##

        return maskBlue

    ## https://stackoverflow.com/questions/51229126/how-to-find-the-red-color-regions-using-opencv ##
    ## https://stackoverflow.com/questions/32522989/opencv-better-detection-of-red-color ##
    low1 = np.array([0, 70, 50], np.uint8)
    high1 = np.array([10, 255, 255], np.uint8)
    mask1 = cv2.inRange(hsv, low1, high1)

    low2 = np.array([170, 70, 50], np.uint8)
    high2 = np.array([180, 255, 255], np.uint8)
    mask2 = cv2.inRange(hsv, low2, high2)
    ## - ##

    maskRed = cv2.add(mask1, mask2)
    return maskRed


def detector(path):
    try:
        os.makedirs('resultado_imgs')
    except FileExistsError:
        print("Ya existe el directorio")
        shutil.rmtree('resultado_imgs')
        os.makedirs('resultado_imgs')
        # pass

    imagenes = os.listdir(path)
    direccion = path + "/"
    total = 0
    corte = 0
    for img in imagenes:
        imgOriginal = cv2.imread(direccion + img, 1)

        imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        imgSalida = cv2.equalizeHist(imgGray)  # Aumenta contraste
        imgSalida = cv2.blur(imgSalida, (3, 3))  # Elimina ruido

        mser = cv2.MSER_create(delta=3, min_area=250, max_area=20000, max_variation=0.1, min_diversity=25)
        regiones, boundingBoxes = mser.detectRegions(imgSalida)

        numDeteccion = 0

        boxCuadrado = [(x, y, w, h) for (x, y, w, h) in boundingBoxes if (
                    w / h <= 1.2 and w / h >= 0.8)]  # Eliminar regiones con proporcion distinta de 1.0 -> se queda con cuadrados practicamente

        listaX = []

        for box in boxCuadrado:
            x, y, w, h = box
            total += 1

            if not contenido(listaX, box):
                listaX.append(box)
                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (255, 0, 0), 2)  # dibuja cada cuadro
                cv2.putText(imgOriginal, str(numDeteccion), (x + 10, y + 10), 0, 0.3, (0, 255, 0), 1)
                if (x >= 10 and y >= 10 and x + w + 20 <= imgOriginal.shape[0] and y + h + 20 <= imgOriginal.shape[
                    1]):  # Mirar desbordamiento imagen para ampliar
                    x -= 10
                    y -= 10
                    w += 20
                    h += 20
                    numDeteccion += 1
                    imgResultado = imgOriginal[y:y + h, x:x + w]

                    porcentaje, tipo = comparacion(imgResultado)

                    if (
                            porcentaje > 0.15):  # p > 0.181 -> 165; p > 0 -> 888; p > 0.2 -> 61 ; p > 0.1 -> 779; p > 0.15 -> 593
                        corte += 1
                        nombre = img[0:5] + "_" + str(numDeteccion) + img[5:]
                        # cv2.imwrite("resultado_imgs/" + nombre, imgResultado)

                        print(tipo, "->", porcentaje)

        cv2.imwrite("resultado_imgs/" + img, imgOriginal)

        # plt.imshow (imgOriginal)
        # plt.show()
        print(img, " analizado")

    print("Total", total, "// Corte", corte)


def contenido(boxes, actual):
    for b1 in boxes:
        if (b1 != actual):
            if bb_intersection_over_union(b1, actual) > 0.4:
                return True

    return False


## https://programmerclick.com/article/2819545676/##
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


## - ##

def comparacion(imagen):
    mascaraRojo = HSV(imagen, 1)
    mascaraAzul = HSV(imagen, 2)

    porcRojo, tipoRojo = calculaPorcentaje(mascaraRojo)
    porcAzul, tipoAzul = calculaPorcentaje(mascaraAzul)

    if (porcRojo < porcAzul):
        return porcAzul, tipoAzul
    else:
        return porcRojo, tipoRojo


def calculaPorcentaje(mascara):
    ## https://programmerclick.com/article/81161880156/ ##
    porcentajes = [0, 0, 0, 0, 0, 0]
    prohibicion = cv2.imread("mascaras/prohibicion.jpg", 0)
    porcentajes[0], _ = structural_similarity(mascara, prohibicion, full=True)

    peligro = cv2.imread("mascaras/peligro.jpg", 0)
    porcentajes[1], _ = structural_similarity(mascara, peligro, full=True)

    stop = cv2.imread("mascaras/stop.jpg", 0)
    porcentajes[2], _ = structural_similarity(mascara, stop, full=True)

    dirProhibida = cv2.imread("mascaras/dirProhibida.jpg", 0)
    porcentajes[3], _ = structural_similarity(mascara, dirProhibida, full=True)

    cedaPaso = cv2.imread("mascaras/cedaPaso.jpg", 0)
    porcentajes[4], _ = structural_similarity(mascara, cedaPaso, full=True)

    dirObligatoria = cv2.imread("mascaras/dirObligatoria.jpg", 0)
    porcentajes[5], _ = structural_similarity(mascara, dirObligatoria, full=True)

    maximo = np.amax(porcentajes)
    posicion = np.where(porcentajes == maximo)
    return maximo, posicion[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    main()

    # Load training data

    # Create the detector

    # Load testing data

    # Evaluate sign detections

# main.py --detector --train_path train_jpg --test_path test_alumnos_jpg




