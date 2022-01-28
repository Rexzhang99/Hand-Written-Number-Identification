import numpy as np
import pickle
from scipy.stats import truncnorm

with open("data/pkl/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:

    def __init__(self,
                 network_structure,  # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
                 learning_rate,
                 bias=None
                 ):

        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):

        bias_node = 1 if self.bias else 0
        self.weights_matrices = []

        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index - 1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2,
                                 sd=1,
                                 low=-rad,
                                 upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuple,
        list or ndarray
        """

        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            if self.bias:
                # adding bias node to the end of the 'input'_vector
                in_vector = np.concatenate((in_vector,
                                            [[self.bias]]))
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[layer_index],
                       in_vector)
            out_vector = activation_function(x)
            # the output of one layer is the input of the next one:
            res_vectors.append(out_vector)
            layer_index += 1

        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
        # The input vectors to the various layers
        output_errors = target_vector - out_vector
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index - 1]

            if self.bias and not layer_index == (no_of_layers - 1):
                out_vector = out_vector[:-1, :].copy()

            tmp = output_errors * out_vector * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)

            # if self.bias:
            #    tmp = tmp[:-1,:]

            self.weights_matrices[layer_index - 1] += self.learning_rate * tmp

            output_errors = np.dot(self.weights_matrices[layer_index - 1].T,
                                   output_errors)
            if self.bias:
                output_errors = output_errors[:-1, :]
            layer_index -= 1

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        no_of_layers = len(self.structure)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector,
                                           [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T

        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index - 1],
                       in_vector)
            out_vector = activation_function(x)

            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector,
                                            [[self.bias]]))

            layer_index += 1

        return out_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm


with open("nist_tests_Multiple.csv", "a") as fh_out:
    for layer_one_nodes in [300,400,800]:
        for layer_two_nodes in [300,400,800]:
            print("*", end="")
            ANN = NeuralNetwork(network_structure=[image_pixels, layer_one_nodes, layer_two_nodes, 10],
                                learning_rate=0.1,
                                bias=None)
            for i in range(len(train_imgs)):
                ANN.train(train_imgs[i], train_labels_one_hot[i])  # Train NN
            train_corrects, train_wrongs = ANN.evaluate(train_imgs, train_labels)
            test_corrects, test_wrongs = ANN.evaluate(test_imgs, test_labels)
            outstr = str(layer_one_nodes) + " " + str(layer_two_nodes) + " "
            outstr += str(train_corrects / (train_corrects + train_wrongs)) + " "
            outstr += str(train_wrongs / (train_corrects + train_wrongs)) + " "
            outstr += str(test_corrects / (test_corrects + test_wrongs)) + " "
            outstr += str(test_wrongs / (test_corrects + test_wrongs))

            fh_out.write(outstr + "\n")
            fh_out.flush()



# Train 25*25-80-150-10 NN
ANN = NeuralNetwork(network_structure=[image_pixels, 80,150, 10],
                               learning_rate=0.1,
                               bias=None)  # Initialize Parameter


for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])  # Train NN

# Error Analysis
for i in range(20):
    res = ANN.run(test_imgs[i])
    print(test_labels[i], np.argmax(res), np.max(res))  # Front 20 image posterior probability

corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
print("accuracy train: ", corrects / ( corrects + wrongs))  # Overall train accuracy
corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
print("accuracy: test", corrects / ( corrects + wrongs))  # Overall test accuracy

train_cm = ANN.confusion_matrix(train_imgs, train_labels)
print(train_cm)
test_cm = ANN.confusion_matrix(test_imgs, test_labels)
print(test_cm)

for i in range(10):
    print("digit: ", i, "precision: ", ANN.precision(i, test_cm), "recall: ", ANN.recall(i, test_cm))





ANN = NeuralNetwork(network_structure=[image_pixels, 150, 10],
                               learning_rate=0.07,
                               bias=None)  # Initialize Parameter

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])  # Train NN
