import numpy as np 
import tensorflow as tf 
import random
import matplotlib.pyplot as plt 

train_data = []
train_labels = []

def digit_to_pixel(n):
	if n==0: 
		return [[1]*5+[0],[1]+[0]*3+[1]+[0],[1]+[0]*3+[1]+[0],[1]+[0]*3+[1]+[0],[1]*5+[0], [0]*6] 
	elif n==1:
		return [[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0]*6]
	elif n==2:
		return [[1]*5+[0], [0]*4 + [1]+[0], [1]*5+[0], [1]+[0]*4+[0], [1]*5+[0],[0]*6]
	elif n==3:
		return [[1]*5+[0], [0]*4+[1]+[0], [1]*5+[0], [0]*4+[1]+[0], [1]*5+[0],[0]*6]
	elif n==4:
		return [[1]+[0]*3+[1]+[0], [1]+[0]*3+[1]+[0], [1]*5+[0], [0]*4+[1]+[0],[0]*4+[1]+[0],[0]*6]
	elif n==5:
		return [[1]*5+[0], [1]+[0]*4+[0], [1]*5+[0], [0]*4+[1]+[0], [1]*5+[0],[0]*6]
	elif n==6:
		return [[1]*5+[0], [1]+[0]*4+[0], [1]*5+[0], [1]+[0]*3+[1]+[0], [1]*5+[0],[0]*6]
	elif n==7:
		return [[1]*5+[0], [0]*4+[1]+[0],[0]*4+[1]+[0],[0]*4+[1]+[0],[0]*4+[1]+[0],[0]*6]
	elif n==8:
		return [[1]*5+[0], [1]+[0]*3+[1]+[0], [1]*5+[0], [1]+[0]*3+[1]+[0], [1]*5+[0],[0]*6]
	elif n==9:
		return [[1]*5+[0], [1]+[0]*3+[1]+[0], [1]*5+[0], [0]*4+[1]+[0], [1]*5+[0],[0]*6]

def gen_num(n, is_sum):
	im = []
	des_len = 4
	if is_sum: des_len = 5
	str_n = str(n)
	if len(str_n) < des_len:
		numzeroes = des_len-len(str_n)
		for i in range(numzeroes):
			str_n = "0" + str_n
	for i in str_n:
		if len(im)==0:
			im+=digit_to_pixel(int(i))
		else:
			tmpim = digit_to_pixel(int(i))
			for j in range(6):
				im[j] = im[j] + tmpim[j]

	return im

def gen_add(n,m):
	return gen_num(n, False) + gen_num(m, False)

def gen_data(n):
	for i in range(n):
		x = random.randint(0,10000)
		y = random.randint(0,10000)
		z = x+y
		train_data.append(np.asarray(gen_add(x,y)))
		train_labels.append(np.asarray(gen_num(z, True)))

gen_data(10000)
