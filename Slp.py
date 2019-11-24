import math

class SLP:
    def __init__(self, output_classes, input_features):
        self.output_classes = output_classes
        self.output_layer = []
        for i in output_classes:
            self.output_layer.append(Perceptron(input_features))

    def train(self, training_set, learning_rate=0.5, iterations=3500):
        self.gradient_descent(iterations, learning_rate, training_set)

    def gradient_descent(self, iteration_count, learning_coeff, training_set):
        predictions = []
        for data_point in training_set:
            predictions.append(self.predict(data_point))

        calculate_error()
        back_propogate_error()
        update_weights()
        return None

    def predict(self, data_sample):
        sums = []

        for i in self.output_layer:
            sums.append(i.lin_sum(data_sample))
        print(sums)
        return self.get_softmaxes(sums)

    def get_softmaxes(self, sums):
        softmaxes = []
        
        denominator = 0
        for netj in sums:
            denominator += math.exp(netj)

        for neti in sums:
            softmaxes.append(math.exp(neti)/denominator)

        return softmaxes

    def calculate_error():
        print("TODO")
        return None

    def back_propogate_error():
        print("TODO")
        return None

    def update_weights(self, new_weights):
        print("TODO")
        return None

class Perceptron:
    def __init__(self, input_features):
        self.input_features = input_features
        #weight is 1/512 since the some of the grayscale values can actually exceed the limit of math.exp's representation, we need this later
        self.incoming_weights = [1/512] * len(input_features)

    def lin_sum(self, data_sample):
        total = 0
        for i in range(len(data_sample)):
            total += round(data_sample[i] * self.incoming_weights[i], 4)
            print(total)
        return total
