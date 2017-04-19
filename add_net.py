import tensorflow as tf
import numpy as np 
import random
import matplotlib.pyplot as plt
from add_input import train_data, train_labels, test_data, test_labels

#the input is a list of numpy arrays
batch_size = 256
dataset_size = train_data.shape[0]
test_size = test_data.shape[0]
graph = tf.Graph()

#makes a relu fc layer
def layer(input, W, b):
	return tf.nn.relu(tf.matmul(input, W)+ b)
#makes a sigmoid layer
def siglayer(input, W, b):
	return tf.sigmoid(tf.matmul(input, W)+b)

with graph.as_default():
	train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 12*24))
	train_labelset = tf.placeholder(tf.float32, shape=(batch_size, 6*30))
	#train_dataset_flat = tf.reshape(train_dataset, shape = [batch_size, 12*24])
	#train_labelset_flat = tf.reshape(train_labelset, shape = [batch_size, 6*30])

	test_dataset = tf.constant(test_data)
	test_labelset = tf.constant(test_labels)
	#test_dataset_flat = tf.reshape(test_dataset, shape = [test_size, 12*24])
	#test_labelset_flat = tf.reshape(test_labels, shape = [test_size, 6*30])

	#layer 1
	W1 = tf.Variable(tf.truncated_normal([12*24, 256], stddev = .01))
	b1 = tf.Variable(tf.truncated_normal([256], stddev = .15))
	h1 = layer(train_dataset, W1, b1)

	#layer 2
	W2 = tf.Variable(tf.truncated_normal([256, 256], stddev = .01))
	b2 = tf.Variable(tf.truncated_normal([256], stddev = .15))
	h2 = layer(h1, W2, b2)

	#layer 3
	W3 = tf.Variable(tf.truncated_normal([256, 256], stddev = .01))
	b3 = tf.Variable(tf.truncated_normal([256], stddev = .15))
	h3 = layer(h2, W3, b3)

	#output layer
	W4 = tf.Variable(tf.truncated_normal([256, 6*30], stddev = .01))
	b4 = tf.Variable(tf.truncated_normal([6*30], stddev = .15))
	h4 = tf.matmul(h3, W4) + b4
	print h4.shape
	out_im = siglayer(h3, W4, b4)
	print out_im.shape

	#define loss and optimizer
	diff = out_im - train_labelset
	sq_diff = tf.multiply(diff,diff)
	sum_sq_diff = tf.reduce_sum(sq_diff, axis = 1)
	loss = tf.reduce_mean(sum_sq_diff)
	optimizer = tf.train.MomentumOptimizer(learning_rate=0.2, momentum=0.9).minimize(loss)

	test_out = siglayer(layer(layer(layer(test_dataset,W1,b1),W2,b2),W3,b3),W4,b4)
	diff_test = test_out - test_labelset
	test_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(diff_test, diff_test), axis = 1))
	test_out = tf.reshape(test_out, shape = [test_size, 6, 30])

num_steps = 30000

with tf.Session(graph = graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")
	for step in range(num_steps):
		offset = (step*batch_size) % (dataset_size - batch_size)
		#generate minibatch
		batch_data = train_data[offset:(offset + batch_size)]
		batch_labels = train_labels[offset:(offset + batch_size)]
		#make dictionary to plug into run
		feed_dict = {train_dataset:batch_data, train_labelset:batch_labels}
		_, l, tl, to = session.run([optimizer, loss, test_loss, test_out],
												 feed_dict = feed_dict)
		if (step%500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Test loss at step %d: %f" % (step, tl))
		if step == num_steps -1:
			#let's see if the test images are computed correctly!
			for i in range(1):
				x = random.randint(0, test_labels.shape[0])
				#plt.show(plt.imshow(test_data[x]))
				#plt.show(plt.imshow(to[x]))