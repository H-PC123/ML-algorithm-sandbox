import argparse
import Slp

implemented_learners = ['naive_bayes', 'support_vector_machine', 'multinomial_perceptron', 'multilayer_perceptron', 'convolutional_neural_net']

parser = argparse.ArgumentParser(description='Runs k-fold cross validation on the input training set using the chosen learners, and report back the performance predictions.')

parser.add_argument('--learners', '-l', action='append', help='Use to enable input learners for testing.')

args = parser.parse_args()

print(args)

for learner in args.learners:
    if learner in implemented_learners:
        __import__(learner)
