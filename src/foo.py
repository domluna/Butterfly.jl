import tensorflow as tf
import time
import numpy as np

w1 = tf.Variable(tf.random_normal((784, 100)))
w2 = tf.Variable(tf.random_normal((100, 1)))
b1 = tf.Variable(tf.zeros(100))
b2 = tf.Variable(tf.zeros(1))

x_ = tf.placeholder(tf.float32, (None, 784))
y_ = tf.placeholder(tf.float32, (None))

def nn(x, y):
    x = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    x = tf.matmul(x, w2) + b2
    return tf.reduce_mean(tf.square(x - y))

out = nn(x_, y_)

x = np.random.randn(32, 784)
y = np.random.randn(32)

with tf.device('/cpu:0'), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1 = time.time()
    sess.run(out, {x_: x, y_: y})
    t2 = time.time()
    print("Time taken 1:", (t2 - t1))
    t1 = time.time()
    sess.run(out, {x_: x, y_: y})
    t2 = time.time()
    print("Time taken 2:", (t2 - t1))
