import mnist
import Mlp
#import os

mnist.temporary_dir = lambda: './MNIST/'

train_images = mnist.train_images().reshape(len(mnist.train_images()), 784)
train_labels = mnist.train_labels()

test_images = mnist.test_images().reshape(len(mnist.test_images()), 784)
test_labels = mnist.test_labels()

#Mnist parser check, make sure everything works
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

mySlp = Mlp.MLP([x for x in range(0, 10)], [28], [x for x in range(97, 97+ 784)])

print(mySlp.predict(train_images[0]))
print(train_labels[0])

print("============ Single Layer Perceptron Testing ===========\n")
#initial test to confirm correctness
pre_eval_metrics = mySlp.test(test_images[:100], test_labels[:100])
print(str(pre_eval_metrics))
#training, uses a portion of data, enough to get decent results and still not take longer than 10 minutes (around 2 to 5)
mySlp.train(train_images, train_labels, iterations=1000)
#test over larger set of test images, enough to get a decent confusion matrix for actual metric calculation
eval_metrics = mySlp.test(test_images[:100], test_labels[:100])
print(eval_metrics)
