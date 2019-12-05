from sklearn import svm
#pretty much just a wrapper class for sklearn's svm class

class Sklearn_SVM:
    def __init__(self):
        self.clf = svm.SVC()
        self.output_classes = [x for x in range(10)]
        
    def train(self, train_images, train_labels, iterations=1000):
        self.clf.fit(train_images, train_labels)
    
    def test(self, test_images, test_labels):
        return self.test_n_report(test_images, test_labels)
    
    def test_n_report(self, testing_set, testing_labels):
        #a loop for testing an input test set w its labels
        #prints and returns the resulting evaluation metrics
        
        print("\n====================== TESTING =========================")
        predictions = list(self.clf.predict(testing_set))
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