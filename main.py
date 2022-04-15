import argparse
import os
import cv2
import matplotlib.pyplot as plt


def media(path, nombres):
    suma = cv2.imread(path + "/" + nombres[0] + "/00000.jpg", 1)
    k = 0
    for i in nombres:
        direccion = path + "/" + i
        imgs = os.listdir(direccion)
        for j in imgs:
            ## https://stackoverflow.com/questions/57723968/blending-multiple-images-with-opencv ##
            img = cv2.imread(direccion + "/" + j, 1)  # imagen en BGR
            img = cv2.resize(img, (suma.shape[1], suma.shape[0]), interpolation = cv2.INTER_CUBIC)
            k = k + 1
            sc_w = 1 / (k + 1)
            fr_w = 1 - sc_w
            suma = cv2.addWeighted(suma, fr_w, img, sc_w, 0)
            ## - ##
    plt.imshow(suma)
    plt.show()
    return suma


def cargarMascaras():
    prohibicion = ["00", "01", "02", "03", "04", "05", "07", "08", "09", "10", "15", "16"]
    media(args.train_path, prohibicion)

    peligro = ["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    media(args.train_path, peligro)

    dirProhibida = ["17"]
    media(args.train_path, dirProhibida)

    cedaPaso = ["13"]
    media(args.train_path, cedaPaso)

    dirObligatoria = ["38"]
    media(args.train_path, dirObligatoria)


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




