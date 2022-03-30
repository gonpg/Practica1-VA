import argparse
import os
import cv2
import matplotlib.pyplot as plt


def loadTrainData(path):
    imgs = os.listdir(path)
    for i in imgs:
        if os.path.isdir(path + "/" + i):
            print("error")
        else:
            print(i)
            img = cv2.imread(path + '/' + str(i), 0)
            plt.imshow(img, cmap="gray")
            plt.show()
            # hist_item = cv2.calcHist([img],[0],None,[256],[0,255])
            # plt.plot(hist_item.reshape(256,))


def main():
    loadTrainData(args.train_path)


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
