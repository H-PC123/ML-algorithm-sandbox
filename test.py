import mnist
import Slp

mnist.datasets_url = "./MNIST/"

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
