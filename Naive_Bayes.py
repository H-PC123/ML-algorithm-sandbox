import math
import random

class Multiclass_Naive_Bayes:
    def __init__(self, output_classes):
       #initializes the learner with the output classes as specified in the list of output classes
       self.output_classes = output_classes
       self.prior_probabilities = [0] * len(output_classes)
       #784! sample space is wayyyyy too much for my poor laptop to calculate in any reasonable amount of time, instead, we split the 28x28 grid into a 3x3 grid, and evaluate if the sum of gray scale values in any of the grids exceeds some value, and mark the sample as true if so
       #28 also doesnt evenly divide by 3 so we're gonna ignore that last row and column lol
       #this is actually a horrible heuristic but im just using it as proof of concept so i guess its fine
       self._posterior_probabilities = []

    def train(self, train_images, train_labels):
        #calculates and saves all the prior and posterior probabilities
        prior_counts = [0] * len(self.output_classes)
        for i in train_labels:
            prior_counts[i] += 1

        self.prior_probabilities = [x/len(train_labels) for x in prior_counts]

        #essentially we reformulate the training set down to 3x3s rather than 28x28 so that we hae a easier to manage event space
        grayscale_sums = []
        for i in range(len(train_images)):
            sample_grayscale_sums = []
            for row in range(3):
                for col in range(3):
                    sample_grayscale_sums.append(self.get_gs_sum(train_images[i], row, col))
            grayscale_sums.append(sample_grayscale_sums)

        posterior_counts = [[0] * pow(2,9) for x in range(len(self.output_classes))]
        for image_index in range(len(grayscale_sums)):
            train_image_sample = grayscale_sums[image_index]
            event_index = 0
            for m_exp in range(len(train_image_sample)):
                if train_image_sample[m_exp] > 574:
                    event_index += pow(2, len(train_image_sample) - m_exp - 1)
            posterior_counts[train_labels[image_index]][event_index] += 1
        
        self.posterior_probabilities = [[y/sum(posterior_counts[x]) for y in posterior_counts[x]] for x in range(len(posterior_counts))]
        return None

    def test(self, test_images, test_labels):
        #tests the input test images in serial
        #passes the labels along to the evaulation methods to evaulate performance
        predictions = []

        for i in range(len(test_labels)):
            numerator_list = []
            current_image_3x3 = self.get_3x3_matrix(test_images[i])
            event_index = self.get_event_index(current_image_3x3)
            for class_index in range(len(self.prior_probabilities)):
                numerator_list.append(self.posterior_probabilities[class_index][event_index] * self.prior_probabilities[class_index])
            maximum_index = self.get_max_bayes(numerator_list)
            predictions.append(self.output_classes[maximum_index])
        self.evaluate_performance(test_labels, predictions)
        return None


    def get_gs_sum(self, image, row_index, col_index):
        #sums the 9x9 square's gray scale values with its top left corner at the row_index,col_index position

        row_index_start = row_index * 9
        col_index_start = col_index * 9
        gs_sum = 0
        matrix_image = image.reshape(28,28)
        for i in range(9):
            for j in range(9):
                gs_sum += matrix_image[row_index_start + i][col_index_start + j]

        return gs_sum

    def get_3x3_matrix(self, image, threshold=574):
        #returns the imput image as a 3x3 0 1 matrix
        #each element is 1 if over the threshold and 0 otherwise
        
        image_3x3 = []
        for i in range(3):
            for j in range(3):
                if self.get_gs_sum(image, i, j) > threshold:
                    image_3x3.append(1)
                else:
                    image_3x3.append(0)
        return image_3x3

    def get_event_index(self, image_3x3):
        #takes a list of 9 0 or 1 values and returns the bigendian deimal representation of the list
        
        event_index = 0
        for i in range(len(image_3x3)):
            event_index += pow(image_3x3[i] * 2, len(image_3x3) - i -1)
        return event_index

    def get_max_bayes(self, numerator_list):
        #calculates all the P(hi|E) and returns the index of maximum
        maximum_prob = 0
        max_index = 0
        for i in range(len(numerator_list)):
            current_prob = numerator_list[i] / sum(numerator_list) if sum(numerator_list) != 0 else 0
            (maximum_prob, max_index) = (current_prob, i) if current_prob > maximum_prob else (maximum_prob, max_index)

        return max_index

    def evaluate_performance(self, test_labels, predictions):
        print("accuracy:")
        tp_count = 0
        for i in range(len(test_labels)):
            tp_count += 1 if test_labels[i] == predictions[i] else 0
        print(tp_count/len(predictions))

        return None


#class Naive_Bayes:
 #   def __init__(self, input_features):

