#!/usr/bin/python

#import libs
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#one_hot list is list which is 0 in most dimensions and 1 in a single dimension
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

#want to multiply the weight against img vectors to produce 10 dimensional vecotrs
W = tf.Variable(tf.zeros([784,10]))
#bias
b = tf.Variable(tf.zeros([10]))
#only ten possible things a given img can be 0-9

#define our model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#define what it means for model to be good
#cost/loss - what it means to be bad model

#cross-entrophy to calculate loss
#y is predicted probability distribution
#y_ is true distribution
#cross-entrophy roughly measures inefficient prediction are for describing truth

y_ = tf.placeholder(tf.float32,[None,10])

#tf.log computes log of each element of y
#multiply each element of y_ w/ tf.log(y)
#tf.reduce_sum adds 2nd dimension elements of y
#tf.reduce_mean computes mean over examples
cross-entrophy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

#minimize cross-entrophy w/ learning rate of .5
train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross-entrophy)

#initialize the variables we created
init = tf.initialize_all_variables()

#launch the model in a Session, and now we run the operation that initializes the variables
sess = tf.Session()
sess.run(init)
