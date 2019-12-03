import Naive_Bayes
import mnist

mnist.temporary_dir = lambda: './MNIST/'

train_images = mnist.train_images().reshape(len(mnist.train_images()), 784)
train_labels = mnist.train_labels()

test_images = mnist.test_images().reshape(len(mnist.test_images()), 784)
test_labels = mnist.test_labels()

myNB = Naive_Bayes.Multiclass_Naive_Bayes([x for x in range(10)])

myNB.train(train_images, train_labels)

post_train_eval = myNB.test(test_images, test_labels)
