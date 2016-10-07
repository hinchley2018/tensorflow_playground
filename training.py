#!/usr/bin/python

#import libs
import tensorflow as tf
import numpy as np

#create 100 phony x,y data points in numpy
x_data = np.random.rand(100).astype(np.float32)
#y = x * .1 +.3
y_data = x_data *.1 + .3

#try to find values for W and b that compute y_data =W* x_data +b
#W should be .1 and b .3
#tensorflow will figure that out
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y= W * x_data + b

#minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(.5)
train = optimizer.minimize(loss)

#before starting init vars
init = tf.initialize_all_variables()

#launch the graph
sess = tf.Session()
sess.run(init)
#Fit the line
for step in range(1000):
    sess.run(train)
    if step % 20 == 0:
        print step,sess.run(W),sess.run(b)
