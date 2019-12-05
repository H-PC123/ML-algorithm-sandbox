import math
import random


class MLP:
    def __init__(self, output_classes, hidden_layers_spec, input_features):
        #output_classes is a list of possible classes for data
        #hidden_layers is a list of the hidden layer sizes, the leftmost will take input from the features and so forth, the element at each index specifies the number of hidden nodes
        #input_features is a list of features for data
        #this initalizer creates a perceptron for each output class, each one with input_features for it's features list

        self.output_classes = output_classes

        self.hidden_layers = []
        previous_layer_size = len(input_features)
        for layer_size in hidden_layers_spec:
            current_hidden_layer = []
            for i in range(layer_size):
                current_hidden_layer.append(Perceptron([x for x in range(previous_layer_size)]))
            previous_layer_size = layer_size
            self.hidden_layers.append(current_hidden_layer)

        self.output_layer = []
        for i in output_classes:
            self.output_layer.append(Perceptron([x for x in range(previous_layer_size)]))

    def train(self, training_set, training_labels, learning_rate=0.0001, iterations=3500):
        #takes the min of the training set size or iterations to run the training (each data point is an iteration
        #performs gradient descent with the cross entropy as the loss function
        #will need to do stochastic sampling cuz this take far too long i legit dont have time to let this thing train
        #TODO stochastically define a sample to use as the training set rather than actually use the whole input training set
        #perhaps another day
        self.gradient_descent(iterations, learning_rate, training_set, training_labels)

    def test(self, testing_set, testing_labels):
        #takes a test set and its corresponding labels as input and prints the accuracy ,precision and recall evaulation metrics. Also returns these values in a list
        #returned list values are:
        #   0) accuracy
        #   1) micro precision
        #   2) micro recall
        #   3) macro precision
        #   4) macro recall
        #   5) per class precision
        #   6) per class recall
        return self.test_n_report(testing_set, testing_labels)

    def gradient_descent(self, iteration_count, learning_rate, training_data, training_labels):
        # a basic implementation of gradient descent over each value of the training data or the iteration_count if that is lower instead
        #each sample updates the weight by a factor of the learning rate, the cross entropy derivative and the original weight
        for i in range(min(iteration_count, len(training_data))):
            if i%(min(iteration_count, len(training_data))/10) == 0:
                print("{:.3f}".format(i/min(iteration_count, len(training_data)) * 100) + "%")

            prediction_sums, prediction_softmaxes = self.predict(training_data[i], mode='train')

            x_entropy = self.calculate_error(training_labels[i], prediction_softmaxes)

            #we start with back propagation for the output layer
            #need to multiply each element of del_errors[i] by sigmoid of prediction_sums[-2][i] after
            delta_outputs = self.back_propagate_output_layer(prediction_softmaxes, training_labels[i])
            #self.update_weights(self.output_layer,learning_rate, delta_outputs, prediction_sums[-2])

            #next we do the the hidden layers
            layer_dels = [delta_outputs]
            for j in range(len(self.hidden_layers)):
                current_layer_sigmoids = [self.get_sigmoid(x) for x in prediction_sums[-2 - j]]
                forward_layer = self.hidden_layers[-1 - j] if j > 0 else self.output_layer
                current_layer_deltas = self.back_propagate_hidden(current_layer_sigmoids, layer_dels[-1 - j], forward_layer)
                layer_dels.insert(0, current_layer_deltas)

            #Update the weights
            previous_layer_sigmoids = [self.get_sigmoid(x) for x in prediction_sums[-2]]
            self.update_weights(self.output_layer, learning_rate, layer_dels[-1], previous_layer_sigmoids)
            for j in range(len(self.hidden_layers)):
                if j != len(self.hidden_layers) - 1:
                    previous_layer_sigmoids = [self.get_sigmoid(x) for x in prediction_sums[-3 - j]]
                else:
                    #not really sigmoids but I shouldve named it input_vector lol
                    previous_layer_sigmoids = prediction_sums[-3 - j]
                self.update_weights(self.hidden_layers[-1 - j], learning_rate, layer_dels[-2 - j], previous_layer_sigmoids)
        return None

    def predict(self, data_sample, mode='test'):
        #uses the model's current weights to evaluate the given data sample
        
        previous_layer_out = [x/512 for x in data_sample]
        layer_sums = [previous_layer_out]
        for hidden_layer in self.hidden_layers:
            current_layer_sums = [x.lin_sum(previous_layer_out) for x in hidden_layer]
            layer_sums.append(current_layer_sums)
            previous_layer_out = [self.get_sigmoid(x) for x in current_layer_sums]

        output_sums = [x.lin_sum(previous_layer_out) for x in self.output_layer]
        layer_sums.append(output_sums)
        softmaxes = self.get_softmaxes(output_sums)
        if mode == 'test':
            return self.output_classes[softmaxes.index(max(softmaxes))]
        elif mode == 'train':
            return layer_sums, softmaxes

    def get_sigmoid(self, s):
        #returns the sigmoid value of the input value s
        return (1/(1 + math.exp(-s)))

    def get_softmaxes(self, sums):
        #takes the weighted sums on a previous data sample and produes the softmax values of each, returns a list of the softmax values
        softmaxes = []
        
        denominator = 0
        for netj in sums:
            denominator += math.exp(netj)

        for neti in sums:
            softmaxes.append(math.exp(neti)/denominator)

        return softmaxes

    def calculate_error(self, target_class, softmaxes):
        #calculates the cross entropy value of the softmaxes and the actual labels
        #could technically be any satisfactory loss function but I've chosen cross entropy since its what we saw in class
        return self.get_cross_entropy(target_class, softmaxes)

    def get_cross_entropy(self, target_class, softmaxes):
        x_entropy = - softmaxes[self.output_classes.index(target_class)]
        return x_entropy

    def back_propagate_output_layer(self, output_data, target):
        #returns the factor the weights should be adjusted by (before a multiplication by the input)
        #should be processed a touch further by a following function
        del_error = []
        #make your life easier, swap these indicies
        for j in range(len(self.output_classes)):
            if j == target:
                del_error.append((output_data[j] - 1)*output_data[j]*(1 - output_data[j]))
            else:
                del_error.append((output_data[j])*output_data[j]*(1 - output_data[j]))
        return del_error

    def back_propagate_hidden(self, outputs, forward_layer_deltas, forward_layer):
        #returns the deltas for this layer of perceptrons, will need to be nultiplied by the feature value and the learning rate before subtracting from the current weight
        deltas = []
        for i in range(len(outputs)):
            deltas.append(sum([(forward_layer_deltas[x]) * forward_layer[x].incoming_weights[i] for x in range(len(forward_layer))]) * outputs[i] * (1 - outputs[i]))

        return deltas

    def update_weights(self, perceptron_list, learning_rate, del_errors, input_values):
        #updates the weights of the perceptrons from the learning rate and the del_errors
        
        for i in range(len(perceptron_list)):
            weight_delta = [learning_rate * del_errors[i] * input_values[x] for x in range(len(input_values))]
            perceptron_list[i].incoming_weights = [perceptron_list[i].incoming_weights[x] - weight_delta[x] for x in range(len(weight_delta))]
        return None

    def test_n_report(self, testing_set, testing_labels):
        #a loop for testing an input test set w its labels
        #prints and returns the resulting evaluation metrics
        predictions = []
        print("\n====================== TESTING =========================")
        for test_sample in testing_set:
            predictions.append(self.predict(test_sample))
        cm = self.get_confusion_matrix(predictions, testing_labels)
        eval_metrics = self.show_performance(cm)
        print("============ END OF TESTING ON " + str(len(testing_labels)) + " SAMPLES =============\n")

        return eval_metrics


    def get_confusion_matrix(self, predictions, test_labels):
        #generates the confusion matrix for the given labels and predictions
        #useful later for display performance metrics
        #output is a 2D array, where the element at index i,j, is the number of times the model predicted i while the true value was j

        cm = [[0]*len(self.output_classes) for i in range(len(self.output_classes))]
        
        for i in range(len(predictions)):
            cm[predictions[i]][test_labels[i]] += 1
        return cm

    def show_performance(self, cm):
        #calculates and displays performance metrix from the confusion matrix

        tp_list = [cm[x][x] for x in range(len(cm))]
        fn_list = [sum(cm[x]) - tp_list[x] for x in range(len(cm))]
        fp_list = [sum([cm[x][y] for x in range(len(cm))]) - tp_list[y] for y in range(len(cm))]
        tn_value = sum([sum(cm[x]) for x in range(len(cm))]) - sum(tp_list)

        accuracy = sum(tp_list)/(sum(fn_list + tp_list))
        (per_class_precisions, per_class_recalls) = self.get_class_metrics(tp_list, fn_list, fp_list)
        (micro_avg_precision, micro_avg_recall) = self.get_micro_avg(tp_list, fn_list, fp_list)
        (macro_avg_precision, macro_avg_recall) = self.get_macro_avg(per_class_precisions, per_class_recalls)

        print("Accuracy : " + str(accuracy))
        print("Micro average precision : " + str(micro_avg_precision))
        print("Micro average recall : " + str(micro_avg_recall))
        print("Per class precisions : " + str(per_class_precisions))
        print("Per class recalls : " + str(per_class_recalls))
        print("Macro average precision : " + str(macro_avg_precision))
        print("Macro average recall : " + str(macro_avg_recall))

        return (accuracy, micro_avg_precision, micro_avg_recall, macro_avg_precision, macro_avg_recall, per_class_precisions, per_class_recalls)

    def get_micro_avg(self, tp_list, fn_list, fp_list):
        #returns a tuple (x, y) where x is the micro average precision and y is the micro average recall
        precision = sum(tp_list)/(sum(tp_list + fp_list))
        recall = sum(tp_list)/(sum(tp_list + fn_list))
        return (precision, recall)

    def get_class_metrics(self, tp_list, fn_list, fp_list):
        #calculates the class precisions and recalls
        #returns a tuple (x, y) where x is the list of per class precisions and y is the list of per class recalls
        precisions = [tp_list[x]/(tp_list[x]+fp_list[x]) if fp_list[x] > 0 else 0 for x in range(len(tp_list))]
        recalls = [tp_list[x]/(tp_list[x]+fn_list[x]) if fn_list[x] > 0 else 0 for x in range(len(tp_list))]
        return (precisions, recalls)
        

    def get_macro_avg(self, precisions, recalls):
        #returns the macro averages in a tuple (precision, recall)
        return (sum(precisions)/len(precisions), sum(recalls)/len(recalls))
        



class Perceptron:
    def __init__(self, input_features):
        self.input_features = input_features
        self.incoming_weights = [round(random.random(), 4) for x in range(len(input_features))]

    def lin_sum(self, data_sample, norm_coeff=1):
        total = 0
        for i in range(len(data_sample)):
            total += data_sample[i] * norm_coeff * self.incoming_weights[i]
        return total
