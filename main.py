#imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

import tensorflow as tf
import numpy as np
import pickle
import gzip
import zipfile as zzip
import os

#*************************************Logistic Regression: Start********************************************
#reading input data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#hyperparameters for logistic regression
iterations = 20000
batch_size = 100
learning_rate = 0.5

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(iterations):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
print('Logistic Regression accuracy on MNIST:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#USPS
filename="proj3_images.zip"

#Defining height,width for resizing the images to 28x28 like MNIST digits
height=28
width=28

#Defining path for extracting dataset zip file
extract_path = "usps_data"

#Defining image,label list
images = []
img_list = []
labels = []

#Extracting given dataset file    
with zzip.ZipFile(filename, 'r') as zippi:
    zippi.extractall(extract_path)

#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            labels.append(int(path[-1]))
            images.append(os.path.join(*path, file)) 

#Resizing images like MNIST dataset   
for idx, imgs in enumerate(images):
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_list.append(img_data)

#Storing image and labels in arrays to be used for training   
USPS_img_array = np.array(img_list)
USPS_img_array = np.subtract(255, USPS_img_array)
USPS_label_array = np.array(labels)

nb_classes = 10
targets = np.array(USPS_label_array).reshape(-1)
aa = np.eye(nb_classes)[targets]
USPS_label_array = np.array(aa, dtype=np.int32)


#convert to tensor
USPS_img_array_tensor=tf.convert_to_tensor(USPS_img_array)
USPS_label_array_tensor=tf.convert_to_tensor(USPS_label_array)


print('Logistic Regression accuracy on USPS:', sess.run(accuracy, feed_dict={x: USPS_img_array, y_: USPS_label_array}))


#*************************************Logistic Regression: End********************************************
#*************************************Single Layer Neural Network: Start**********************************

class Network(object):

    def __init__(self, sizes):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None,usps_data=None):
      
        if test_data: n_test = len(test_data)
        if usps_data: n_usps = len(usps_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch MNIST {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch MNIST{0} complete".format(j))
            
            
            
            if usps_data:
                print("Epoch USPS {0}: {1} / {2}".format(
                    j, self.evaluate(usps_data), n_usps))
            else:
                print("Epoch USPS {0} complete".format(j))
        print("accuracy for MNIST achieved is {0}.".format(self.evaluate(test_data)/n_test))
        print("accuracy for USPS achieved is {0}.".format(self.evaluate(usps_data)/n_usps))

           
        
    def update_mini_batch(self, mini_batch, eta):
       
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
      
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
       
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


hidden_layer_neurons = 30
epochs = 30
learning_rate = 3.0
mini_batch_size = 10


filename="proj3_images.zip"
#Defining height,width for resizing the images to 28x28 like MNIST digits
height=28
width=28
#Defining path for extracting dataset zip file
extract_path = "usps_data"
#Defining image,label list
images = []
img_list = []
labels = []
#Extracting given dataset file    
with zzip.ZipFile(filename, 'r') as zippy:
    zippy.extractall(extract_path)
#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            labels.append(int(path[-1]))
            images.append(os.path.join(*path, file)) 
#Resizing images like MNIST dataset   
for idx, imgs in enumerate(images):
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_list.append(img_data)
    
#Storing image and labels in arrays to be used for training   
USPS_img_array = np.array(img_list)
USPS_img_array = np.subtract(255, USPS_img_array)
USPS_label_array = np.array(labels)
#print(USPS_label_array.shape)
nb_classes = 10
targets = np.array(USPS_label_array).reshape(-1)
aa = np.eye(nb_classes)[targets]
USPS_label_array = np.array(aa, dtype=np.int32) 

testInputs = [np.reshape(x, (784, 1)) for x in USPS_img_array]


f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
tr_d, va_d, te_d = u.load()
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
testData = list(zip(testInputs, vectorized_result(USPS_label_array)))

training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))
validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
validation_data = list(zip(validation_inputs, va_d[1]))
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = list(zip(test_inputs, te_d[1]))

net = Network([784, hidden_layer_neurons, 10])
net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data, usps_data= testData)



#*************************************Single Layer Neural Network: End*****************************************
#*************************************Convolutional Neural Network: Start**************************************

def cnn_model_fn(features, labels, mode):
  
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    
  #load USPS data as MNIST array
  filename="proj3_images.zip"

  #Defining height,width for resizing the images to 28x28 like MNIST digits
  height=28
  width=28

  #Defining path for extracting dataset zip file
  extract_path = "usps_data"

  #Defining image,label list
  images = []
  img_list = []
  labels = []

  #Extracting given dataset file    
  with zzip.ZipFile(filename, 'r') as zipper:
    zipper.extractall(extract_path)

  #Extracting labels,images array needed for training    
  for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            labels.append(int(path[-1]))
            images.append(os.path.join(*path, file)) 

  #Resizing images like MNIST dataset   
  for idx, imgs in enumerate(images):
      img = Image.open(imgs).convert('L') 
      img = img.resize((height, width), Image.ANTIALIAS)
      img_data = list(img.getdata())
      img_list.append(img_data)

  #Storing image and labels in arrays to be used for training   
  USPS_img_array = np.array(img_list)
  USPS_img_array = np.subtract(255, USPS_img_array)
  USPS_label_array = np.array(labels)

  nb_classes = 10
  targets = np.array(USPS_label_array).reshape(-1)
  aa = np.eye(nb_classes)[targets]
  USPS_label_array = np.array(aa, dtype=np.int32)  
    
    
    
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data_mnist = mnist.train.images # Returns np.array
  train_labels_mnist = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  mnist_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data_mnist},
    y=train_labels_mnist,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn,steps=500, hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  #eval_results_mnist = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results_mnist)
  
  #calculating for USPS dataset
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": USPS_img_array},
    y=USPS_label_array,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn,steps=500, hooks=[logging_hook])
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results_usps = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results_usps)
  

#tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
  tf.app.run()

#*************************************Convolutional Neural Network: End********************************************

