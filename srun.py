from skimage.feature import local_binary_pattern, hog
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn import linear_model
import sklearn.ensemble
import time
import os


def multi_feature(imagename):
    hog1 = hog_feature(imagename)
    lbp1 = lbp(imagename, block=10)
    return np.hstack([hog1, lbp1])

def hog_feature(imagename):
    image = cv2.imread(imagename)  # read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
    image = cv2.resize(image, (128,128), interpolation=cv2.INTER_LINEAR)
    hog1 = hog(image, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")

    return hog1  # the characteristic vertex is an array holding 256 * block ** 2 elements


def lbp(imagename, block=5, radius=1,
        n_points=8):  # input: the filename(str) & number of blocks(int)
    image = cv2.imread(imagename)  # read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
    image = cv2.resize(image, (100,100), interpolation=cv2.INTER_LINEAR)
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
    print('F1:', 2 * TP / (2 * TP + FP + FN) , "\n")  # print F1 score
    return 2 * TP / (2 * TP + FP + FN)


def fake_train_test_score(train_label, train_histogram, test_label,
                          test_histogram):
    # 弃用 ##################################
    # 输入：test_histogram含有3600个histogram（对应3600张图片的特征向量），test_label含3600个1或-1（对应3600张图片的标签）#

    print("SVM starts")
    svc = SVC(kernel='linear', degree=2, gamma=1, coef0=0, verbose=1)
    time_start = time.time()
    svc.fit(train_histogram, train_label.ravel())  # 训练#
    print("Training Time: {}".format(time.time() - time_start))
    time_start = time.time()
    predict_result = svc.predict(test_histogram)  # 测试，predict含有400个1或-1#
    print("Testing Time: {}".format(time.time() - time_start))
    score(predict_result)

    print("AdaBoost starts")
    ada_b = sklearn.ensemble.AdaBoostClassifier(
        n_estimators=10000, learning_rate=0.01)
    time_start = time.time()
    ada_b.fit(train_histogram, train_label.ravel())
    print("Training Time: {}".format(time.time() - time_start))
    time_start = time.time()
    predict_result = ada_b.predict(test_histogram)
    print("Testing Time: {}".format(time.time() - time_start))
    score(predict_result)


def train_test_score(model, train_label, train_histogram, test_label,
                        test_histogram):
    print("Training starts")
    time_start = time.time()
    model.fit(train_histogram, train_label.ravel())
    print("Training Time: {}".format(time.time() - time_start))
    time_start = time.time()
    predict_result = model.predict(test_histogram)
    print("Testing Time: {}".format(time.time() - time_start))
    return score(predict_result)


if __name__ == '__main__':
    list_of_label = []
    list_of_hist = []  # generates the two sets of labels/histograms at once and then use 10-fold cross validation
    # There is no need to generate the training sets and test sets repeatedly. TAs just set a bad example for us.
    start_time = time.time()
    photos_done_count = 0  # can be ignored, just used to show how many photos have been processed with lbp

    for number_of_sets in range(10):
        print("[{}] Generating HOG vector...".format(number_of_sets))
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
                    hog_feature("dataset/" + str(number_of_sets) + "/" +
                        line.rstrip('\n').split(' ')[0]) for line in faces
                ])
                np.savez('cache/' + str(number_of_sets) + ".npz", train_label,
                         train_hist)
        list_of_label.append(train_label)
        list_of_hist.append(train_hist)
        print("[{}] Generating HOG vector... done {}".format(
            number_of_sets, "(cached)" if isCached else ""))
    # lbp finishes

    print("\nCollecting Data Done. Time used: {}\n".format(time.time() - start_time))

    scores = []
    for time_of_test in range(10):
        train_hist = np.vstack(
            [list_of_hist[i] for i in range(10) if i != time_of_test])
        train_label = np.vstack(
            [list_of_label[i] for i in range(10) if i != time_of_test])
        test_hist = list_of_hist[time_of_test]
        test_label = list_of_label[time_of_test]
        print("Test time {}".format(time_of_test))
        # model = sklearn.ensemble.GradientBoostingClassifier(
        #     n_estimators=20000, verbose=1, max_features=2,
        #     subsample=0.5)  # verbose=1 will display the progress
        model = linear_model.LogisticRegression(solver="liblinear", C=3, intercept_scaling=3)
        # model = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_features=100, n_jobs=-1, verbose=1)
        scores.append(train_test_score(model, train_label, train_hist, test_label, test_hist))
    
    print("Average F1: {}".format(sum(scores)/len(scores)))
