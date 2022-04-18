import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def media(path, nombres):
    suma = cv2.imread(path + "/" + nombres[0] + "/00000.jpg", 1)
    k = 0
    for i in nombres:
        direccion = path + "/" + i
        imgs = os.listdir(direccion)
        for j in imgs:
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
        plt.imshow(maskBlue)
        plt.show()
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

def cargarMascaras():
    prohibicion = ["00", "01", "02", "03", "04", "05", "07", "08", "09", "10", "15", "16"]
    resultado = HSV(media(args.train_path, prohibicion), 1)
    cv2.imwrite("prohibicion.jpg", resultado)

    peligro = ["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    resultado = HSV(media(args.train_path, peligro), 1)
    cv2.imwrite("peligro.jpg", resultado)

    dirProhibida = ["17"]
    resultado = HSV(media(args.train_path, dirProhibida), 1)
    cv2.imwrite("dirProhibida.jpg", resultado)

    cedaPaso = ["13"]
    resultado = HSV(media(args.train_path, cedaPaso), 1)
    cv2.imwrite("cedaPaso.jpg", resultado)

    dirObligatoria = ["38"]
    resultado = HSV(media(args.train_path, dirObligatoria), 2)
    cv2.imwrite("dirObligatoria.jpg", resultado)

def main():
    cargarMascaras()
    # hacer comparacion, calculo regiones, etc

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




