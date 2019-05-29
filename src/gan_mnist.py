# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


#Sample noisy mini-batch to the generator
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


#Xavier Initialization  
def xavier_init(size):
    N_val = size[0]
    xavier_std_dev = 1. / tf.sqrt(N_val / 2.)
    return tf.random_normal(shape=size, stddev=xavier_std_dev)

  
#Discriminator parameters initialization
D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
param_D = [D_W1, D_W2, D_b1, D_b2]

#Generator parameters initialization
G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))
param_G = [G_W1, G_W2, G_b1, G_b2]
  
  

#Generator network
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

#Discriminator network
def discriminator(x,no):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    if no == 2:
      D_logit = tf.matmul(D_h1, D_W2) + (D_b2 + len(np.unique(x)))
      length = len(np.unique(x))
      print(length)
    else:
      D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

#Visualizing the generator sample output during training
def display(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
  
  
#Tensors for real data and sample noise  
X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])
  


#Prediction probability and raw logit outputs from discriminator
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X,1)
D_fake, D_logit_fake = discriminator(G_sample,2)



#LOSS Functions

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -D_loss

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

#Optimizer selection for mini-batch SGD trianing
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=param_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=param_G)


#read data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


#mini-batch size and sample noise size 
minibatch_size = 128
Z_dim = 100

#Initialize Sessions and train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('gen_sample/'):
    os.makedirs('gen_sample/')

i = 0

for iteration in range(1000000):
    if iteration % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = display(samples)
        plt.savefig('gen_sample/mnist_{}.png'.format(str(i).zfill(4)), bbox_inches='tight')
        i += 1
        plt.close(fig)


    X_minibatch, _ = mnist.train.next_batch(minibatch_size)

    _, D_loss_iter = sess.run([D_solver, D_loss], feed_dict={X: X_minibatch, Z: sample_Z(minibatch_size, Z_dim)})
    _, G_loss_iter = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(minibatch_size, Z_dim)})

    if iteration % 1000 == 0:
        print('Iter: {}'.format(iteration))
        print('D loss: {:.4}'. format(D_loss_iter))
        print('G_loss: {:.4}'.format(G_loss_iter))
        print()



