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

    def train(self, train_images, train_labels, iterations = 6000):
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
        print("\n====================== TESTING =========================")

        predictions = []

        for i in range(len(test_labels)):
            numerator_list = []
            current_image_3x3 = self.get_3x3_matrix(test_images[i])
            event_index = self.get_event_index(current_image_3x3)
            for class_index in range(len(self.prior_probabilities)):
                numerator_list.append(self.posterior_probabilities[class_index][event_index] * self.prior_probabilities[class_index])
            maximum_index = self.get_max_bayes(numerator_list)
            predictions.append(self.output_classes[maximum_index])
        evaluation_metrics = self.evaluate_performance(test_labels, predictions)

        print("============ END OF TESTING ON " + str(len(test_labels)) + " SAMPLES =============\n")
        return evaluation_metrics


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
        cm = self.get_confusion_matrix(predictions, test_labels)

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

    def get_confusion_matrix(self, predictions, test_labels):
        #generates the confusion matrix for the given labels and predictions
        #useful later for display performance metrics
        #output is a 2D array, where the element at index i,j, is the number of times the model predicted i while the true value was j

        cm = [[0]*10 for i in range(len(self.output_classes))]
        
        for i in range(len(predictions)):
            cm[predictions[i]][test_labels[i]] += 1
        return cm

    def get_micro_avg(self, tp_list, fn_list, fp_list):
        #returns a tuple (x, y) where x is the micro average precision and y is the micro average recall
        precision = sum(tp_list)/(sum(tp_list + fp_list))
        recall = sum(tp_list)/(sum(tp_list + fn_list))
        return (precision, recall)

    def get_macro_avg(self, precisions, recalls):
        #returns the macro averages in a tuple (precision, recall)
        return (sum(precisions)/len(precisions), sum(recalls)/len(recalls))

    def get_class_metrics(self, tp_list, fn_list, fp_list):
        #calculates the class precisions and recalls
        #returns a tuple (x, y) where x is the list of per class precisions and y is the list of per class recalls
        precisions = [tp_list[x]/(tp_list[x]+fp_list[x]) if fp_list[x] > 0 else 0 for x in range(len(tp_list))]
        recalls = [tp_list[x]/(tp_list[x]+fn_list[x]) if fn_list[x] > 0 else 0 for x in range(len(tp_list))]
        return (precisions, recalls)

