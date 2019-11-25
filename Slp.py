import math
import random


class SLP:
    def __init__(self, output_classes, input_features):
        self.output_classes = output_classes
        self.output_layer = []
        for i in output_classes:
            self.output_layer.append(Perceptron(input_features))

    def train(self, training_set, training_labels, learning_rate=0.5, iterations=3500):
        self.gradient_descent(iterations, learning_rate, training_set, training_labels)

    def test(self, testing_set, testing_labels):
        self.test_n_report(testing_set, testing_labels)

    def gradient_descent(self, iteration_count, learning_rate, training_data, training_labels):
        for i in range(min(iteration_count, len(training_data))):
            print("{:.3f}".format(i/min(iteration_count, len(training_data)) * 100) + "%")
            prediction_sums = self.get_sums(training_data[i])
            prediction_softmaxes = self.get_softmaxes(prediction_sums)

            x_entropy = self.calculate_error(training_labels[i], prediction_softmaxes)

            del_errors = self.back_propogate_error(prediction_softmaxes, training_labels[i], training_data[i])
            self.update_weights(learning_rate, del_errors)

        return None

    def predict(self, data_sample):
        sums = self.get_sums(data_sample)
        softmaxes = self.get_softmaxes(sums)
        return self.output_classes[softmaxes.index(max(softmaxes))]

    def get_sums(self, data_sample):
        sums = []
        for i in self.output_layer:
            sums.append(i.lin_sum(data_sample))
        return sums

    def get_softmaxes(self, sums):
        softmaxes = []
        
        denominator = 0
        for netj in sums:
            denominator += math.exp(netj)

        for neti in sums:
            softmaxes.append(math.exp(neti)/denominator)

        return softmaxes

    def calculate_error(self, target_class, softmaxes):
        return self.get_cross_entropy(target_class, softmaxes)

    def get_cross_entropy(self, target_class, softmaxes):
        x_entropy = - softmaxes[self.output_classes.index(target_class)]
        return x_entropy

    def back_propogate_error(self, softmaxes, target, data_sample):
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
        for i in range(len(self.output_layer)):
            weight_delta = [learning_rate * x for x in del_errors[i]]
            self.output_layer[i].incoming_weights = [self.output_layer[i].incoming_weights[x] - weight_delta[x] for x in range(len(weight_delta))]
        return None

    def test_n_report(self, testing_set, testing_labels):
        predictions = []
        for test_sample in testing_set:
            predictions.append(self.predict(test_sample))

        tp_count = 0
        for i in range(len(predictions)):
            if predictions[i] == testing_labels[i]:
                tp_count += 1

        print("accuracy : " + str(tp_count/len(predictions)))



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
