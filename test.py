import mnist
import Slp
#import os

mnist.temporary_dir = lambda: './MNIST/'

train_images = mnist.train_images().reshape(len(mnist.train_images()), 784)
train_labels = mnist.train_labels()

test_images = mnist.test_images().reshape(len(mnist.test_images()), 784)
test_labels = mnist.test_labels()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

mySlp = Slp.SLP([x for x in range(0, 10)], [x for x in range(97, 97+ 784)])

print(mySlp.predict(train_images[0]))
print(train_labels[0])

print("===== Gradient Descent Iteration Test =====\n")

mySlp.test(test_images[:100], test_labels[:100])
mySlp.train(train_images, train_labels, iterations=100000)
mySlp.test(test_images[:100], test_labels[:100])
