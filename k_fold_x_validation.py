import mnist
import numpy
import random
import json

import Slp
import Naive_Bayes
import Mlp
import Sklearn_svm
import Sklearn_gnb
import Sklearn_mlp

fold_amount = 10


def split_dataset(train_images, train_labels, fold_count):
    #returns the dataset split into evenly into the proper amount of folds
    #data should be shuffled before being entered into this function

    subset_len = int(len(train_images)/fold_count)

    images_subsets = [list(train_images[x * subset_len:(x + 1) * subset_len]) for x in range(fold_count)]
    labels_subsets = [list(train_labels[x * subset_len:(x + 1) * subset_len]) for x in range(fold_count)]

    return (images_subsets, labels_subsets)

def scramble_dataset(train_images, train_labels):
    #shuffles the training images and labels (like so that the labels are still correctly matched obviously)
    #can be reused multiple times for more scrambledness
    
    myList = list(zip(train_images, train_labels))
    random.shuffle(myList)
    (a, b) = zip(*myList)
    return (numpy.array(a), numpy.array(b))

def initialize_learners():
    #learners to be tested should be entered into this array

    learners = []

    #multinomial perceptron learner
    learners.append(Slp.SLP([x for x in range(0, 10)], [x for x in range(97, 97 + 784)]))

    #naive bayes learner
    learners.append(Naive_Bayes.Multiclass_Naive_Bayes([x for x in range(10)]))

    #multi layer perceptron learner !!DOESNT WORK ATM
    #learners.append(Mlp.MLP([x for x in range(0,10)], [28], [x for x in range(97, 97 + 784)]))
    
    #scikit learn's SVM inplementation
    learners.append(Sklearn_svm.Sklearn_SVM())
    
    #scikit learn's Gaussian Naive Bayes implementation
    learners.append(Sklearn_gnb.Sklearn_Naive_Bayes())
    
    #sklearn's MLP implementation, default parameters
    learners.append(Sklearn_mlp.Sklearn_MLP())

    return learners

def get_train_test_sets(fold_number, image_folds, label_folds):
    (fold_test_images, fold_test_labels) = (image_folds.pop(fold_number), label_folds.pop(fold_number))
    return (fold_test_images, fold_test_labels)

def k_fold_x_validation(folds):
    #performs the k fold cross validation using the learners specified in the initialize_learners method
    #images and labels contain the training set before shuffling and splitting into the proper number of folds

    #import mnist training set
    (train_images, train_labels) = (mnist.train_images().reshape(len(mnist.train_images()), 784)[:1000], mnist.train_labels()[:1000])

    #shuffle the training data
    (train_images, train_labels) = (train_images, train_labels) = scramble_dataset(train_images, train_labels)

    #split up the mnist data into k "folds"
    (split_images, split_labels) = split_dataset(train_images, train_labels, 10)

    #perform the cross validation for each learner, record the results in the results array
    #each ith entry in the array contains a list of the learners' performance evaulations on the ith folds
    #performance evaulations are stored as a list of 7 values:
        #0  accuracy
        #1  micro averaged precision
        #2  micro averaged recall
        #3  macro averaged precision
        #4  macro averaged recall
        #5  per class precisions (list)
        #6  per class recalls (list)
    #We additionally append the list of learner classes to the end of the list so we have a reference for the indices
    performance_evaluations = {}

    fold_length = int(len(train_labels)/folds)
    for k in range(folds):
        print("===== Using " + str(k + 1) + "th out of " + str(folds) + " folds as test set for " + str(folds) +  "-fold cross-validation =====")
        #initializing as we need new learners for each evaluation
        learners = initialize_learners()
        learner_evaluations= {}

        for learner in learners:
            print("    Training " + str(learner) + " learner:")
            (fold_test_images, fold_test_labels) = (split_images[k], split_labels[k])
            for i in range(folds):
                #skips over the test set
                if not i == k:
                    learner.train(split_images[i], split_labels[i], iterations=len(split_labels[i]))

            #the test functions should give us the evaulation metrics of the test in the earlier specified format (as a tuple)
            print("     Testing the trained model on the test images:")
            learner_evaluation = learner.test(fold_test_images, fold_test_labels)
            learner_evaluations[str(type(learner))] = (learner_evaluation)
        performance_evaluations[k] = (learner_evaluations)
    
    return performance_evaluations

evaluations = k_fold_x_validation(10)


#also calculate the averages, this is more or less what we actually want

evaluation_sums = {}
evaluation_averages = {}
for learner in evaluations[0]:
    evaluation_sums[learner] = [0] * 5
    evaluation_averages[learner] = [0] * 5

for fold in evaluations:
    for learner in evaluations[fold]:
        for metric_index in range(5):
            evaluation_sums[learner][metric_index] += evaluations[fold][learner][metric_index]
            
for learner_sums in evaluation_sums:
    evaluation_averages[learner_sums] = [x/10 for x in evaluation_sums[learner_sums]]

#write the results to file (really do not want to lose this), the cross validation takes forever lol
#we take the average over the folds using the first 5 metrics (not the per class evaluations)
        #0  accuracy
        #1  micro averaged precision
        #2  micro averaged recall
        #3  macro averaged precision
        #4  macro averaged recall
json.dump(evaluation_averages, open('x_validation_out.txt', 'w'))