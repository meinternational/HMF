import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

tf_f = dict()
tf_f['tensorflow.nn.relu'] = tf.nn.relu
tf_f['tensorflow.nn.sigmoid'] = tf.nn.sigmoid
tf_f['tensorflow.abs'] = tf.abs
tf_f['tensorflow.identity'] = tf.identity
tf_f['tensorflow.tanh'] = tf.tanh
tf_f['tensorflow.nn.softplus'] = tf.nn.softplus

class Base(object):

	def __init__(self, settings):
		self.settings = settings
		self.placeholder = self.build_placeholder()
		self.model = self.build_model()
		self.optimizer = ScipyOptimizerInterface(self.model['cost'], method='L-BFGS-B', options={'maxiter': self.settings['max_iter'], 'disp': True})
		init = tf.global_variables_initializer()
		self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
		self.sess.run(init)

	def g(self, h, cost='l1'):
		if cost=='logcosh':
			return tf.reduce_sum( 0.5 * tf.log(tf.cosh(2*h)))

		if cost=='l1':
			return tf.reduce_sum(tf.abs(h))

		if cost=='l1_approx':
			return tf.reduce_sum(tf.sqrt(tf.constant(10e-8) + tf.pow(h,2)))

		if cost=='exp':
			return tf.reduce_sum(-tf.exp(-tf.pow(h,2)/2.0))

	def kl_divergence(self,x, y): # Kullback-Leibler divergence
		return tf.log( x ) - tf.log( y ) + ( y / x ) - 1.0


	def filter2toeplitz(self, conv_filter): # converts filter to Toeplitz matrix
		len_h = conv_filter.get_shape().as_list()[0]
		toplitz = list()
		for t in range(self.settings['n_input']):
			if t==0:
				toplitz.append(tf.concat([conv_filter, tf.zeros(self.settings['n_input']-t-len_h)],0))

			elif t>0 and (t+len_h<self.settings['n_input']):
				toplitz.append(tf.concat([tf.zeros(t), conv_filter, tf.zeros(self.settings['n_input']-t-len_h)],0))

			else:
				toplitz.append(tf.concat([tf.zeros(t), conv_filter[0:self.settings['n_input']-t]],0))

		H = tf.transpose(tf.reshape(tf.concat(toplitz,0),(self.settings['n_input'],self.settings['n_input'])))
		return H

	def toeplitz2filter(self, H): # converts Toeplitz matrix to filter
		hw = self.settings['filter_length']/2
		hrft = list()
		for i,c in enumerate(range(hw,self.settings['n_input']-(hw+1))):
			hrft.append( H[c-hw:c+hw,c] )

		return tf.add_n(hrft)/(float(len(hrft)))

	def build_model(self): # overwrite this function
		pass

	def build_placeholder(self):
		placeholder = dict()
		placeholder['x'] = tf.placeholder(tf.float32)
		placeholder['GM'] = tf.placeholder(tf.float32)
		placeholder['MASK'] = tf.placeholder(tf.float32)
		if 'filter_length' in self.settings:
			placeholder['t'] = tf.placeholder(tf.float32, [self.settings['filter_length']])

		return placeholder

	def fit(self, input_dict):
		feed_dict = dict()
		for key in list(self.placeholder.keys()):
			feed_dict[self.placeholder[key]] = input_dict[key]

		self.optimizer.minimize(self.sess, feed_dict=feed_dict)
		return self

	def get_params(self, input_dict):
		feed_dict = dict()
		for key in list(self.placeholder.keys()):
			feed_dict[self.placeholder[key]] = input_dict[key]

		out = self.sess.run(self.model, feed_dict=feed_dict)
		self.sess.close()
		tf.reset_default_graph()
		return out

class CanonicalHRFMatrixFactorizationFast(Base):

		def __init__(self, settings):
			super(CanonicalHRFMatrixFactorizationFast, self).__init__(settings)

		def build_model(self):
			model = dict()
			######################### VARS ###################
			model['b1'] = tf.Variable(tf.zeros([self.settings['n_hidden'], 1], dtype=tf.float32), trainable=self.settings['train_b1'])
			model['neural'] = tf.nn.l2_normalize(tf_f[self.settings['f(neural)']](tf.get_variable("neural", dtype=tf.float32, shape=[
													self.settings['n_hidden'], self.settings['n_input']], initializer=tf.contrib.layers.xavier_initializer())), 1) 

			model['hrf'] = tf.Variable(self.settings['hrfi'], dtype=tf.float32, trainable=self.settings['train_hrf'])
			model['b2'] = tf.Variable(tf.zeros([1, 1], dtype=tf.float32), trainable=self.settings['train_b2'])
			######################## MODEL ###################
			H = self.filter2toeplitz(model['hrf'])
			model['bold'] = tf.transpose(tf.matmul(H, tf.transpose(model['neural'])))
			model['h'] = self.placeholder['MASK'] * tf_f[self.settings['f(WX)']](tf.matmul(model['bold'], self.placeholder['x']) + model['b1'])
			######################## COST ###################
			model['l2'] = tf.nn.l2_loss(self.placeholder['GM'] * tf.subtract((tf.matmul(tf.transpose(model['bold']), model['h']) + model['b2']), self.placeholder['x'])) / (self.settings['n_feature'])
			cost = list()
			cost.append(model['l2'])

			if self.settings['lambda1'] > 0.0:
				model['rho_hat'] = tf.reduce_sum(model['h'], axis=1, keepdims=True) / self.settings['n_feature']
				model['kl_space'] = tf.reduce_sum(self.kl_divergence(1.0/model['rho_hat'], 1.0/self.settings['mu']))
				model['lambda1_c'] = self.settings['lambda1'] * model['kl_space']
				cost.append(model['lambda1_c'])
			
			if self.settings['lambda2'] > 0.0:
				model['lambda2_c'] = self.settings['lambda2'] * tf.reduce_sum(self.g(
					model['neural'][:, 1:] - model['neural'][:, :-1], self.settings['g'])) / self.settings['n_input']
				cost.append(model['lambda2_c'])

			if self.settings['lambda3'] > 0.0:
				MASK3D = tf.reshape(self.placeholder['MASK'], self.settings['dims']+(1,))
				H3D = tf.reshape(tf.transpose(model['h']), self.settings['dims']+(self.settings['n_hidden'],))

				model['lambda3_c'] = self.settings['lambda3'] * tf.reduce_sum(self.g(MASK3D[1:, :, :, :]*(H3D[1:, :, :, :] - H3D[:-1, :, :, :]), self.settings['g'])) / self.settings['n_feature'] \
									+ self.settings['lambda3'] * tf.reduce_sum(self.g(MASK3D[:, 1:, :, :]*(H3D[:, 1:, :, :] - H3D[:, :-1, :, :]), self.settings['g'])) / self.settings['n_feature'] \
									+ self.settings['lambda3'] * tf.reduce_sum(self.g(MASK3D[:, :, 1:, :]*(H3D[:, :, 1:, :] - H3D[:, :, :-1, :]), self.settings['g'])) / self.settings['n_feature']

				cost.append(model['lambda3_c'])

			model['cost'] = tf.add_n(cost)

			return model