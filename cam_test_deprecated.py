import cv2
import joblib
from skimage.feature import local_binary_pattern
import numpy as np
import sklearn.ensemble
import sys


def lbp(image, block=5, radius=1, n_points=8):  # input: the filename(str) & number of blocks(int)
    width, height = image.shape[1], image.shape[0]  # read the resolution of the image
    column, row = width // block, height // block  # calculate how many pixels each block have
    hist = np.array([])
    for i in range(block * block):
        lbp1 = local_binary_pattern(
            image[row * (i // block):row * ((i // block) + 1), column * (i % block):column * ((i % block) + 1)], n_points, radius,
            'nri_uniform')
        hist1, _ = np.histogram(
            lbp1, density=True, bins=256, range=(0, 256))  # convert to histogram
        hist = np.concatenate((hist, hist1))

    return hist  # the characteristic vertex is an array holding 256 * block ** 2 elements


cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gb = joblib.load(sys.argv[1])

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detects = classifier.detectMultiScale(gray, 1.3, 5)
    for d in detects:
        (x, y, w, h) = d
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop, (60, 60), cv2.INTER_LINEAR)
        vec = lbp(crop).reshape(1, -1)
        result = gb.predict(vec)
        print(result)
        color = (0, 0, 255) if result == 0 else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
    cv2.imshow("im", img)
    cv2.waitKey(10)

