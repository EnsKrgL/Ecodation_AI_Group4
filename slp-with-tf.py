# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 21:59:16 2022

@author: ITU
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

import tensorflow as tf
import matplotlib.pyplot as plt

#parametreler
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

x = tf.placeholder("float", [None, 784]) #data
y = tf.placeholder("float", [None, 10]) #labels

#model yaratma

#model ağırlıklarını belirleme
W = tf.Variable(tf.zeros([784, 10])) #weights/ağırlıklar
b = tf.Variable(tf.zeros([10])) #bias

#modelimizi inşa etme
activation = tf.nn.softmax(tf.matmul(x, W) + b)

#cross_entropy kullanarak hatayı minimize etme
cross_entropy = y * tf.log(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Çizim ayarları
avg_set = []
epoch_set = []

#değişkenlerin ilk atamalarını yapma işlemi
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            
            
        #her bir epoch için logları gösterme
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost:","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
        
    print("Eğitim fazı tamamlandı")
    
    plt.plot(epoch_set, avg_set, 'o', label = 'Logistic Regression Eğitim Fazı')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #Test model
    dogru_tahmin = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    #accuracy(doğruluk) hesabı
    accuracy = tf.reduce_mean(tf.cast(dogru_tahmin, "float"))
    print("Model accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    



















