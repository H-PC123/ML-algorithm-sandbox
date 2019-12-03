import math
import random


class SLP:
    def __init__(self, output_classes, input_features):
        #output_classes is a list of possible classes for data
        #input_features is a list of features for data
        #this initalizer creates a perceptron for each output class, each one with input_features for it's features list

        self.output_classes = output_classes
        self.output_layer = []
        for i in output_classes:
            self.output_layer.append(Perceptron(input_features))

    def train(self, training_set, training_labels, learning_rate=0.5, iterations=3500):
        #takes the min of the training set size or iterations to run the training (each data point is an iteration
        #performs gradient descent with the cross entropy as the loss function
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
            prediction_sums = self.get_sums(training_data[i])
            prediction_softmaxes = self.get_softmaxes(prediction_sums)

            x_entropy = self.calculate_error(training_labels[i], prediction_softmaxes)

            del_errors = self.back_propogate_error(prediction_softmaxes, training_labels[i], training_data[i])
            self.update_weights(learning_rate, del_errors)

        return None

    def predict(self, data_sample):
        #uses the model's current weights to evaluate the given data sample
        sums = self.get_sums(data_sample)
        softmaxes = self.get_softmaxes(sums)
        return self.output_classes[softmaxes.index(max(softmaxes))]

    def get_sums(self, data_sample):
        #produces the sums from the perceptrons and returns them in a list for further processing
        sums = []
        for i in self.output_layer:
            sums.append(i.lin_sum(data_sample))
        return sums

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

    def back_propogate_error(self, softmaxes, target, data_sample):
        #returns the factor the weights should be adjusted by 
        #should be processed a touch further by a following function
        del_error = []
        #make your life easier, swap these indicies
        for j in range(len(self.output_classes)):
            del_error_feature_i = []
            for i in range(len(data_sample)):
                if j == target:
                    del_error_feature_i.append((data_sample[i]/512)*(softmaxes[j] - 1))
                else:
                    del_error_feature_i.append((data_sample[i]/512)*(softmaxes[j]))
            del_error.append(del_error_feature_i)
        return del_error

    def update_weights(self, learning_rate, del_errors):
        #updates the weights of the perceptrons from the learning rate and the del_errors
        for i in range(len(self.output_layer)):
            weight_delta = [learning_rate * x for x in del_errors[i]]
            self.output_layer[i].incoming_weights = [self.output_layer[i].incoming_weights[x] - weight_delta[x] for x in range(len(weight_delta))]
        return None

    def test_n_report(self, testing_set, testing_labels):
        #a loop for testing an input test set w its labels
        #prints and returns the resulting evaluation metrics
        predictions = []
        softmaxes = []
        print("\n====================== TESTING =========================")
        for test_sample in testing_set:
            temp_sums = self.get_sums(test_sample)
            softmaxes.append(self.get_softmaxes(temp_sums))
            predictions.append(self.output_classes[softmaxes[-1].index(max(softmaxes[-1]))])
        cm = self.get_confusion_matrix(predictions, testing_labels)
        eval_metrics = self.show_performance(cm)
        print("============ END OF TESTING ON " + str(len(testing_labels)) + " SAMPLES =============\n")

        return eval_metrics


    def get_confusion_matrix(self, predictions, test_labels):
        #generates the confusion matrix for the given labels and predictions
        #useful later for display performance metrics
        #output is a 2D array, where the element at index i,j, is the number of times the model predicted i while the true value was j

        cm = [[0]*10 for i in range(len(self.output_classes))]
        
        for i in range(len(predictions)):
            cm[predictions[i]][test_labels[i]] += 1
        return cm

    def show_performance(self, cm):
        #calculates and dosplays performance metrix from the confusion matrix

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
        #weight is ~1/512 since the some of the grayscale values can actually exceed the limit of math.exp's representation, we need this later
        self.incoming_weights = [round(random.random(), 4) for x in range(len(input_features))]

    def lin_sum(self, data_sample, norm_coeff=1/512):
        total = 0
        for i in range(len(data_sample)):
            total += data_sample[i] * norm_coeff * self.incoming_weights[i]
        return total
