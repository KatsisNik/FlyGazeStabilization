# Setting Warnings and Info off for clearer output on terminal
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


#Global variables

#Number of inputs for the Halters
dh_up = 100
dh_down = 10
dh = dh_up + dh_down
#Number of inputs for the Compound Eyes
dc_up = 100
dc_down = 10
dc = dc_up + dc_down
#Number of inputs for the Motor Neurons
dn_up = 2-1
dn_down = 10
dn = dn_up + dn_down

# Creating some experimental data
T_i = 100 	#time of 1 experiment (in seconds)
T_s = 0.01	#sampling intervals (in seconds)
time = np.arange(0, T_i, T_s)
frequencies = [0.10, 0.15,  0.27,  0.45,  0.74,  1.22,  2.02,  3.33,  5.50]
gains 	   =  [1.00, 1.00,  1.00,  0.99,  0.99,  0.98,  0.98,  0.97,  0.90]
phases 	   =  [0.10, 0.30,  0.60,  0.80,  1.00,  2.00,  3.00,  4.00,  5.00]
TRL, HRL = [], []
for f,g,ph in zip(frequencies, gains, phases):
	TRL.append((np.sin(2*np.pi*f*time + ph).astype(np.float32)))
	HRL.append((g*np.sin(2*np.pi*f*time).astype(np.float32)).tolist())
TRM_ALL = np.array(TRL) + 0.00*np.random.randn(*np.array(TRL).shape)
HRM_ALL = np.array(HRL) + 0.01*np.random.randn(*np.array(HRL).shape)







############ Making Computational Graph ############
tf.reset_default_graph()
xh = tf.placeholder(tf.float32, (dh, None))
wh = tf.get_variable('wh', [dh, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
xc = tf.placeholder(tf.float32, (dc, None))
wc = tf.get_variable('wc', [dc, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
oh = tf.matmul(tf.transpose(wh), xh)
oc = tf.matmul(tf.transpose(wc), xc)
xn = tf.placeholder(tf.float32, (dn, None))
wa = tf.get_variable('wa', [1, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
wn = tf.get_variable('wn', [dn, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
hr = wa*(oh + oc) + tf.matmul(tf.transpose(wn), xn)
hr_true = tf.placeholder(tf.float32, (1, None))
reg_term = tf.norm(wh)**2 + tf.norm(wc)**2 + tf.norm(wa)**2 + tf.norm(wn)**2
cost = tf.reduce_mean(1/2*(hr - hr_true)**2) + 0.01*reg_term
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(cost)
# optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.01).minimize(cost)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
init = tf.global_variables_initializer()
#####################################################





# Making initializations
dmax = np.max([dh, dc, dn])
TRM = np.zeros((len(frequencies), dmax))
OHM = np.zeros((len(frequencies), dmax))
OCM = np.zeros((len(frequencies), dmax))
HRM = np.zeros((len(frequencies), dmax))
TRM = np.concatenate((TRM, TRM_ALL[:, [0]]), axis = 1)
XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T
list_of_costs = []




#Starting our session
sess = tf.Session()
sess.run(init)


#Training
for i in range(len(time)-1):
	if i%100==0:
		print('iteration: ', i, '/', len(time), end = '\r')
	opt, last_cost, last_oh, last_oc, last_hr = sess.run([optimizer, cost, oh, oc, hr], 
		feed_dict = {xh: XH, xc: XC, xn: XN, hr_true: HRM_ALL[:, [i]].T})		
	TRM = np.concatenate((TRM, TRM_ALL[:, [i+1]]), axis = 1)
	OCM = np.concatenate((OCM, last_oc.T), axis = 1)
	OHM = np.concatenate((OHM, last_oh.T), axis = 1)
	HRM = np.concatenate((HRM, last_hr.T), axis = 1)
	list_of_costs.append(last_cost.item())
	XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
	XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
	XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T


# Saving the weights of the NN
WH, WC, Wn, Wa = sess.run([wh, wc, wn, wa])
WN = np.concatenate((Wa, Wn), axis = 0)




# Variables for plots
t_i = 0
t_f = len(time)-1

fig, ax = plt.subplots(9, 1)
for i in range(0,9):
	ind_f = i
	ax[i].plot(time[t_i:t_f], TRM_ALL[ind_f,  t_i:t_f], '--b')
	ax[i].plot(time[t_i:t_f], HRM[ind_f, t_i+dmax: t_f+dmax], 'r')
	ax[i].plot(time[t_i:t_f], HRM_ALL[ind_f, t_i:t_f], 'k')
plt.show()








# Testing phase
input('press Enter for testing phase')

# frequencies =   [0.10, 0.15,  0.27,  0.45,  0.74,  1.22,  2.02,  3.33,  5.50]
# gains 	   =  [1.00, 1.00,  1.00,  0.99,  0.99,  0.98,  0.98,  0.97,  0.90]
# phases 	   =  [0.10, 0.30,  0.60,  0.80,  1.00,  2.00,  3.00,  4.00,  5.00]
TRM_ALL = np.sin(2*np.pi*0.74*time).astype(np.float32).reshape(1, -1) # Input signal
TRM_ALL += np.sin(2*np.pi*1.22*time).astype(np.float32).reshape(1, -1)
HRM_ALL = 0.99*np.sin(2*np.pi*0.74*time- 1.0).astype(np.float32).reshape(1, -1) # Expected true response
HRM_ALL += 0.98*np.sin(2*np.pi*1.22*time - 2.0).astype(np.float32).reshape(1,-1)

TRM_ALL = 0*TRM_ALL
alpha = np.array(range(0,10000))
TRM_ALL[0, alpha] = 1.0 - np.exp(-alpha/500)


TRM = np.zeros((1, dmax))
OHM = np.zeros((1, dmax))
OCM = np.zeros((1, dmax))
HRM = np.zeros((1, dmax))
TRM = np.concatenate((TRM, TRM_ALL[:, [0]]), axis = 1)
XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T



for i in range(len(time)-1):
	if i%100==0:
		print('iteration: ', i, '/', len(time), end = '\r')
	last_oh, last_oc, last_hr = sess.run([oh, oc, hr], 
		feed_dict = {xh: XH, xc: XC, xn: XN, hr_true: HRM_ALL[:, [i]].T})		
	TRM = np.concatenate((TRM, TRM_ALL[:, [i+1]]), axis = 1)
	OCM = np.concatenate((OCM, last_oc.T), axis = 1)
	OHM = np.concatenate((OHM, last_oh.T), axis = 1)
	HRM = np.concatenate((HRM, last_hr.T), axis = 1)
	XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
	XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
	XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T



plt.plot(time, TRM_ALL.squeeze(), '--b')
# plt.plot(time, HRM_ALL.squeeze(), 'k')
plt.plot(time, HRM[0,dmax-1:].squeeze(), 'r')
plt.xlim(0,100)
plt.show()














































































'''
t = np.arange(0, 5, T_s)
TRL = []
HRL = []
for f,g,ph in zip(frequencies, gains, phases):
	TRL += (np.sin(2*np.pi*f*t + ph).astype(np.float32)).tolist()
	HRL += (g*np.sin(2*np.pi*f*t).astype(np.float32)).tolist()

#training set.... (need a validation for hyperparam tuning)
TR = np.array(TRL) + 0.00*np.random.randn(len(TRL))
HR = np.array(HRL) + 0.00*np.random.randn(len(HRL))

TRM_ALL = np.array(TR).reshape(1, -1)
HRM_ALL = np.array(HR).reshape(1, -1)




dmax = np.max([dh, dc, dn])
TRM = np.zeros((1, dmax))
OHM = np.zeros((1, dmax))
OCM = np.zeros((1, dmax))
HRM = np.zeros((1, dmax))

list_of_costs = []

with tf.Session() as sess:
	sess.run(init)
	TRM = np.concatenate((TRM, TRM_ALL[:, [0]]), axis = 1)
	XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
	XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
	XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T

	for i in range(TRM_ALL.shape[1]-1):
		if i%100==0:
			print('iteration: ', i, '/', TRM_ALL.shape[1], end = '\r')
		last_oh, last_oc, last_hr = sess.run([oh, oc, hr], 
			feed_dict = {xh: XH, xc: XC, xn: XN})		
		TRM = np.concatenate((TRM, TRM_ALL[:, [i+1]]), axis = 1)
		OCM = np.concatenate((OCM, last_oc.T), axis = 1)
		OHM = np.concatenate((OHM, last_oh.T), axis = 1)
		HRM = np.concatenate((HRM, last_hr.T), axis = 1)
		XH = np.concatenate((TRM[:, -dh_up:], OHM[:, -dh_down:]), axis = 1).T
		XC = np.concatenate((TRM[:, -dc_up:]-HRM[:, -dc_up:], OCM[:, -dc_down:]), axis = 1).T
		XN = np.concatenate((OCM[:, -dn_up:]+OHM[:, -dn_up:], HRM[:, -dn_down:]), axis = 1).T

	WH = sess.run(wh)
	WC = sess.run(wc)
	Wn = sess.run(wn)
	Wa = sess.run(wa)
	WN = np.concatenate((Wa, Wn), axis = 0)
'''









print('------ OKKKKKKK ------- ')