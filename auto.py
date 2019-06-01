from skimage.feature import local_binary_pattern
import numpy as np
import cv2
from sklearn.svm import SVC
import sklearn.ensemble
import time
import os


def lbp(imagename, block=5, radius=1,
        n_points=8):  # input: the filename(str) & number of blocks(int)
    image = cv2.imread(imagename)  # read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
    width, height = image.shape[1], image.shape[0]  # read the resolution of the image
    column, row = width // block, height // block  # calculate how many pixels each block have
    hist = np.array([])
    for i in range(block * block):
        lbp1 = local_binary_pattern(
            image[row * (i // block):row * ((i // block) + 1), column *
                  (i % block):column * ((i % block) + 1)], n_points, radius,
            'nri_uniform')
        hist1, _ = np.histogram(
            lbp1, density=True, bins=256, range=(0, 256))  # convert to histogram
        hist = np.concatenate((hist, hist1))

    return hist  # the characteristic vertex is an array holding 256 * block ** 2 elements


def score(predict_result):
    # input: the predict result
    # output: None
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(predict_result)):
        if test_label[i] == 1:
            if predict_result[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predict_result[i] == 1:
                FP += 1
            else:
                TN += 1
    print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN', FN)
    print('F1:', 2 * TP / (2 * TP + FP + FN))  # print F1 score
    return 2 * TP / (2 * TP + FP + FN)


def train_test_score_gb(gb, train_label, train_histogram, test_label,
                        test_histogram):
    print("GradientBoosting starts")
    time_start = time.time()
    gb.fit(train_histogram, train_label.ravel())
    print("Training Time: {}".format(time.time() - time_start))
    time_start = time.time()
    predict_result = gb.predict(test_histogram)
    print("Testing Time: {}".format(time.time() - time_start))
    return score(predict_result)


if __name__ == '__main__':
    list_of_label = []
    list_of_hist = []  # generates the two sets of labels/histograms at once and then use 10-fold cross validation
    # There is no need to generate the training sets and test sets repeatedly. TAs just set a bad example for us.
    start_time = time.time()
    photos_done_count = 0  # can be ignored, just used to show how many photos have been processed with lbp

    for number_of_sets in range(10):
        print("[{}] Generating LBP vector...".format(number_of_sets))
        if os.path.isfile("cache/" + str(number_of_sets) + ".npz"):
            isCached = True
            npz_data = np.load("cache/" + str(number_of_sets) + ".npz")
            train_label = npz_data["arr_0"]
            train_hist = npz_data["arr_1"]
        else:
            with open('dataset/' + str(number_of_sets) + '/faces.txt', 'r') as faces:
                isCached = False
                train_label = np.array([[int(line.rstrip('\n').split(' ')[1])] for line in faces])
                faces.seek(0)
                train_hist = np.array([
                    lbp("dataset/" + str(number_of_sets) + "/" +
                        line.rstrip('\n').split(' ')[0]) for line in faces
                ])
                np.savez('cache/' + str(number_of_sets) + ".npz", train_label,
                         train_hist)
        list_of_label.append(train_label)
        list_of_hist.append(train_hist)
        print("[{}] Generating LBP vector... done {}".format(
            number_of_sets, "(cached)" if isCached else ""))
    # lbp finishes

    print("\nCollecting Data Done. Time used: {}".format(time.time() - start_time))
    
    params = range(1, 5)
    a_scores = []
    for param in params: 
        gb = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=param*10, verbose=1, max_features="auto",
            subsample=0.5)  # verbose=1 will display the progress
        scores = []
        for time_of_test in range(10):
            train_hist = np.vstack(
                [list_of_hist[i] for i in range(10) if i != time_of_test])
            train_label = np.vstack(
                [list_of_label[i] for i in range(10) if i != time_of_test])
            test_hist = list_of_hist[time_of_test]
            test_label = list_of_label[time_of_test]
            print("Test time {}".format(time_of_test))
            scores.append(train_test_score_gb(gb, train_label, train_hist, test_label, test_hist))
        print(scores)
        a_score = sum(scores) / len(scores)
        print("\nAverage score: " + str(a_score))
        a_scores.append((param,a_score))

    print(max(a_scores, key=lambda x: x[1]))
