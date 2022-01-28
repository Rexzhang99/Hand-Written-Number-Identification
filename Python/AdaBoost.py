from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from sklearn.ensemble import AdaBoostClassifier as ada
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time

sns.set()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# n=0.2
# x_train=x_train[1:int(60000*n)+1,]
# x_test=x_test[1:int(10000*n)+1,]
# y_train=y_train[1:int(60000*n)+1,]
# y_test=y_test[1:int(10000*n)+1,]

print('Train size:', x_train.shape[0])
print('Test size:', x_test.shape[0])

# n_estimators=300
# with open("Result/nist_tests_ada.csv", "w") as fh_out:
#     for learning_rate in [0.1,0.2,0.5,1]:
#         clf = ada(DecisionTreeClassifier(max_depth=2),n_estimators=n_estimators, learning_rate=learning_rate)
#         print('n_estimators=',n_estimators,'learning_rate=',learning_rate)
#         start_time=time.time()
#         clf_fit = clf.fit(x_train, y_train)
#         training_time = time.time() - start_time
#         print("--- %s seconds ---" % (training_time))
#
#         train_staged_score = np.zeros((n_estimators,))
#         test_staged_score = np.zeros((n_estimators,))
#         for i, score in enumerate(clf_fit.staged_score(x_train, y_train)):
#             train_staged_score[i] = score
#         for i, score in enumerate(clf_fit.staged_score(x_test, y_test)):
#             test_staged_score[i] = score
#
#
#         for stage in range(n_estimators):
#             outstr = str(n_estimators) + " " + str(learning_rate) + " " + str(stage) + " " + str(train_staged_score[stage]) + " " + str(
#                 test_staged_score[stage]) + " " + str(training_time)
#             fh_out.write(outstr + "\n")
#             fh_out.flush()

n_estimators=300
learning_rate=0.1
clf = ada(DecisionTreeClassifier(max_depth=2),n_estimators=n_estimators, learning_rate=learning_rate)
print('n_estimators=',n_estimators,'learning_rate=',learning_rate)
start_time=time.time()
clf_fit = clf.fit(x_train, y_train)
training_time = time.time() - start_time
print("--- %s seconds ---" % (training_time))


with open("Result/nist_tests_ada_final.csv", "w") as fh_out:
    train_staged_score = np.zeros((n_estimators,))
    test_staged_score = np.zeros((n_estimators,))
    for i, score in enumerate(clf_fit.staged_score(x_train, y_train)):
        train_staged_score[i] = score
    for i, score in enumerate(clf_fit.staged_score(x_test, y_test)):
        test_staged_score[i] = score

    for stage in range(n_estimators):
        outstr = str(n_estimators) + " " + str(learning_rate) + " " + str(stage) + " " + str(train_staged_score[stage]) + " " + str(
            test_staged_score[stage]) + " " + str(training_time)
        fh_out.write(outstr + "\n")
        fh_out.flush()

n_estimators=600
learning_rate=0.1
clf = ada(DecisionTreeClassifier(max_depth=2),n_estimators=n_estimators, learning_rate=learning_rate)
print('n_estimators=',n_estimators,'learning_rate=',learning_rate)
start_time=time.time()
clf_fit = clf.fit(x_train, y_train)
training_time = time.time() - start_time
print("--- %s seconds ---" % (training_time))


with open("Result/nist_tests_ada_600.csv", "w") as fh_out:
    train_staged_score = np.zeros((n_estimators,))
    test_staged_score = np.zeros((n_estimators,))
    for i, score in enumerate(clf_fit.staged_score(x_train, y_train)):
        train_staged_score[i] = score
    for i, score in enumerate(clf_fit.staged_score(x_test, y_test)):
        test_staged_score[i] = score

    for stage in range(n_estimators):
        outstr = str(n_estimators) + " " + str(learning_rate) + " " + str(stage) + " " + str(train_staged_score[stage]) + " " + str(
            test_staged_score[stage]) + " " + str(training_time)
        fh_out.write(outstr + "\n")
        fh_out.flush()

confusion_matrix(y_test, clf_fit.predict(x_test))

clf_fit.score(x_test,y_test)