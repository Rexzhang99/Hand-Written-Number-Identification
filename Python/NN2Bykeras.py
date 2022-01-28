import numpy as np
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,SGD
from sklearn.metrics import confusion_matrix
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Train size:', x_train.shape[0])
print('Test size:', x_test.shape[0])

# model = Sequential()
# model.add(Dense(120, input_shape=(784,)))
# model.add(Activation('relu'))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# for l in model.layers:
#     print(l.output.name, l.input_shape, '==>', l.output_shape)
# print(model.summary())
#
# batch_size = 200
# epochs = 1
#
# model.compile(loss='mean_squared_error',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
#
# score = model.evaluate(x_test, y_test, verbose=100)
#
# print('Test loss:', round(score[0], 3))
# print('Test accuracy:', round(score[1], 3))
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# epochs=10
# activation='relu'
# with open("Result/nist_tests_keras_2.csv", "a") as fh_out:
#     for hidden_nodes1 in [20, 50, 100, 150, 250, 500]:
#         for hidden_nodes2 in [20, 50, 100, 150, 250, 500]:
#             for batch_size in [100, 200, 300]:
#                 for loss in ['mean_squared_error','categorical_crossentropy']:
#                     for learning_rate in [0.001,0.01]:
#                         model = Sequential()
#                         model.add(Dense(hidden_nodes1, activation=activation, input_shape=(784,)))
#                         model.add(Dense(hidden_nodes2, activation=activation))
#                         model.add(Dense(num_classes, activation='softmax'))
#                         model.compile(loss=loss,
#                                       optimizer=RMSprop(learning_rate=learning_rate),
#                                       metrics=['accuracy'])
#
#                         for l in model.layers:
#                             print(l.name, l.input_shape, '==>', l.output_shape, 'Activation=', activation)
#                         print('Loss=', loss,'batch size=', batch_size)
#                         print(model.summary())
#
#                         history = model.fit(x_train, y_train,
#                                             batch_size=batch_size,
#                                             epochs=epochs,
#                                             verbose=2,
#                                             validation_data=(x_test, y_test))
#
#
#                         # score = model.evaluate(x_test, y_test, verbose=100)
#                         # print(history.history['accuracy'])
#
#                         for epoch in range(epochs):
#                             outstr = str(hidden_nodes1) + " " +str(hidden_nodes2) + " " + str(activation) + " " + str(batch_size)+ " " + str(loss)+ " " + str(learning_rate)
#                             outstr += " " + str(epoch) + " "+ str(history.history['accuracy'][epoch]) + " "+ str(history.history['val_accuracy'][epoch])
#                             outstr += " "+ str(history.history['loss'][epoch]) + " "+ str(history.history['val_loss'][epoch])
#                             fh_out.write(outstr + "\n")
#                             fh_out.flush()

epochs=10
hidden_nodes1=500
hidden_nodes2=150
activation='relu'

loss='mean_squared_error'
batch_size=200
learning_rate=0.001
model = Sequential()
model.add(Dense(hidden_nodes1, activation=activation, input_shape=(784,)))
model.add(Dense(hidden_nodes2, activation=activation))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='mean_squared_error',
          optimizer=RMSprop(learning_rate=learning_rate),
          metrics=['accuracy'])

for l in model.layers:
    print(l.name, l.input_shape, '==>', l.output_shape, 'Activation=', activation)
print('Loss=', loss)
print('batch size=', batch_size)
print(model.summary())

start_time = time.time()
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(x_test, y_test))
training_time=time.time() - start_time
print("--- %s seconds ---" % (training_time))

def hot_to_cat(y_test):
    decoded_datum = np.zeros((len(y_test), 1), int)
    for i in range(len(y_test)):
        decoded_datum[i,] = np.argmax(y_test[i])
    return decoded_datum


cm=confusion_matrix(hot_to_cat(y_test), model.predict_classes(x_test))
print(cm)