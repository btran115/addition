import tensorflow as tf
import numpy as np 
from add_input import train_data, train_labels

#the input is a list of numpy arrays
batch_size = 100
dataset_size = len(train_data)
graph = tf.Graph()
with graph.as_default():
	train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 12,24))
	train_labelset = tf.placeholder(tf.float32, shape=(batch_size, 6,30))
	train_dataset_flat = tf.reshape(train_dataset, shape = [batch_size, 12*24])
	train_labelset_flat = tf.reshape(train_labelset, shape = [batch_size, 6*30])

	#layer 1
	W1 = tf.Variable(tf.truncated_normal([12*24, 256]))
	b1 = tf.Variable(tf.truncated_normal([256]))
	h1 = tf.nn.relu(tf.matmul(train_dataset_flat, W1) + b1)

	#layer 2
	W2 = tf.Variable(tf.truncated_normal([256, 256]))
	b2 = tf.Variable(tf.truncated_normal([256]))
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

	#layer 3
	W3 = tf.Variable(tf.truncated_normal([256, 256]))
	b3 = tf.Variable(tf.truncated_normal([256]))
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

	#output layer
	W4 = tf.Variable(tf.truncated_normal([256, 6*30]))
	b4 = tf.Variable(tf.truncated_normal([6*30]))
	labels = tf.sigmoid(tf.matmul(h3, W4) + b4)

	loss = tf.reduce_mean(tf.nn.l2_loss(labels - train_labelset_flat))

	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

num_steps = 1

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
		_, l = session.run([optimizer, loss], feed_dict = feed_dict)
		if (step%500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))