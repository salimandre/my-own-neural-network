from myloads import *

class Dense(Layer):
	def __init__(self, input_units, output_units, init_option="default", reg_param = 0., **optim_method):
		"""
		A dense layer is a layer which performs a learned affine transformation:
		f(x) = <W*x> + b
		"""

		# Optimization parameters depending on the method used
		if optim_method["name"]=="SGD":
			self.optim_method_name = "SGD"
			self.learning_rate = optim_method["learning_rate"]
		elif optim_method["name"]=="NAG_method":
			self.optim_method_name = "NAG_method" 
			self.learning_rate = optim_method["learning_rate"]
			self.gamma_momentum = optim_method["gamma_momentum"]
			self.theta_weights_past = np.zeros((input_units, output_units))
			self.theta_biases_past = np.zeros(output_units)
		elif optim_method["name"]=="SGD_with_momentum":
			self.optim_method_name = "SGD_with_momentum" 
			self.learning_rate = optim_method["learning_rate"]
			self.gamma_momentum = optim_method["gamma_momentum"]
			self.grad_weights_past = np.zeros((input_units, output_units))
			self.grad_biases_past = np.zeros(output_units)
		elif optim_method["name"]=="Adagrad":
			self.optim_method_name = "Adagrad"
			self.learning_rate = optim_method["learning_rate"]
			self.squared_grad_weights_accumulation = np.ones((input_units, output_units))
			self.squared_grad_biases_accumulation = np.ones((output_units))
			self.epsilon = 1e-8
		elif optim_method["name"]=="RMSProp":
			self.optim_method_name = "RMSProp"
			self.learning_rate = optim_method["learning_rate"]
			self.gamma = optim_method["gamma"]
			self.squared_grad_weights_accumulation = np.ones((input_units, output_units))
			self.squared_grad_biases_accumulation = np.ones((output_units))
			self.epsilon = 1e-8
		elif optim_method["name"]=="ADAM":
			self.optim_method_name = "ADAM"
			self.learning_rate = optim_method["learning_rate"]
			self.gamma_adaptative = optim_method["gamma_adaptative"]
			self.gamma_momentum = optim_method["gamma_momentum"]
			self.squared_grad_weights_accumulation = np.zeros((input_units, output_units))
			self.squared_grad_biases_accumulation = np.zeros((output_units))
			self.grad_weights_accumulation = np.zeros((input_units, output_units))
			self.grad_biases_accumulation = np.zeros((output_units))
			self.epsilon = 1e-8

		# Regularization parameter
		self.reg_param = reg_param # Here we add a regularization parameter

		# initialize biases at zeros
		self.biases = np.zeros(output_units)

		# initialize weights with normal initialization, or by Xavier's method
		if init_option == 'StandardNormal':
			self.weights = np.random.randn(input_units, output_units)
		if init_option == '0.1StandardNormal':
			self.weights = np.random.randn(input_units, output_units)*0.1
		elif init_option == 'Xavier':
			self.weights = np.random.randn(input_units, output_units)*np.sqrt(1./input_units)
		elif init_option == 'Glorot&Bengio':
			self.weights = np.random.randn(input_units, output_units)*np.sqrt(2./(input_units+output_units))
		elif init_option == 'He&Zhang&Ren&Sun':
			self.weights = np.random.randn(input_units, output_units)*np.sqrt(2./input_units)
		elif init_option == 'default':
			self.weights = np.random.randn(input_units, output_units)*0.01
		
	def forward(self,input):
		"""
		Perform an affine transformation:
		f(x) = <W*x> + b
		
		input shape: [batch, input_units]
		output shape: [batch, output units]
		"""
		return input.dot(self.weights) + self.biases
	
	def backward(self,input,grad_output):
		
		# compute d f / d x = d f / d dense * d dense / d x
		# where d dense/ d x = weights transposed
		grad_input = grad_output.dot(self.weights.T)
		
		# compute gradient for regularization term
		grad_reg = 2.*self.reg_param*self.weights;

		# compute gradient w.r.t. weights and biases
		if self.optim_method_name == "SGD":
			grad_weights = input.T.dot(grad_output) + grad_reg
			grad_biases = np.sum(grad_output, axis=0)
			assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
		
			# Here we perform a stochastic gradient descent step. 
			self.weights = self.weights - self.learning_rate * grad_weights
			self.biases = self.biases - self.learning_rate * grad_biases

		elif self.optim_method_name == "SGD_with_momentum":
			# compute gradient adding an exponentially decreasing sum of past gradients called momentum
			grad_weights_present = input.T.dot(grad_output) + grad_reg
			grad_weights_momentum = self.gamma_momentum * self.grad_weights_past + self.learning_rate * grad_weights_present

			grad_biases_present = np.sum(grad_output, axis=0)
			grad_biases_momentum = self.gamma_momentum * self.grad_biases_past + self.learning_rate * grad_biases_present
			assert grad_weights_momentum.shape == self.weights.shape and grad_biases_momentum.shape == self.biases.shape
			
			# Here we update our stock of past weights and biases gradients
			self.grad_weights_past = grad_weights_momentum
			self.grad_biases_past = grad_biases_momentum

			# Here we perform a stochastic gradient descent step with momentum. 
			self.weights = self.weights - grad_weights_momentum
			self.biases = self.biases - grad_biases_momentum

		elif self.optim_method_name == "NAG_method":
			# compute classical gradient descent step
			grad_weights_present = input.T.dot(grad_output) + grad_reg
			self.theta_weights_present = self.weights - self.learning_rate * grad_weights_present

			grad_biases_present = np.sum(grad_output, axis=0)
			self.theta_biases_present = self.biases - self.learning_rate * grad_biases_present

			# Here we perform the momentum step of SGD with NAG using theta_t and theta_t-1
			self.weights = self.theta_weights_present + self.gamma_momentum * (self.theta_weights_present - self.theta_weights_past)
			self.biases = self.theta_biases_present + self.gamma_momentum * (self.theta_biases_present - self.theta_biases_past)

			# Here we update past thetas 
			self.theta_weights_past = self.theta_weights_present
			self.theta_biases_past = self.theta_biases_present

		elif self.optim_method_name == "Adagrad":
			# compute gradients for weights and biases parameters
			grad_weights = input.T.dot(grad_output) + grad_reg
			grad_biases = np.sum(grad_output, axis=0)

			# compute adaptative learning rates using gradients accumulation
			weights_learning_rate = self.learning_rate * 1./np.sqrt(self.squared_grad_weights_accumulation + self.epsilon)
			biases_learning_rate = self.learning_rate * 1./np.sqrt(self.squared_grad_biases_accumulation + self.epsilon)

			# perform a gradient descent step using adaptative learning rates
			self.weights = self.weights - np.multiply(weights_learning_rate, grad_weights)
			self.biases = self.biases - np.multiply(biases_learning_rate, grad_biases)

			# update of accumulations of squarred gradients
			self.squared_grad_weights_accumulation = self.squared_grad_weights_accumulation + grad_weights**2
			self.squared_grad_biases_accumulation = self.squared_grad_biases_accumulation + grad_biases**2

		elif self.optim_method_name == "RMSProp":
			# compute gradients for weights and biases parameters
			grad_weights = input.T.dot(grad_output) + grad_reg
			grad_biases = np.sum(grad_output, axis=0)

			# compute unbiased moments (towards 0)
			# compute adaptative learning rates using gradients accumulation
			weights_learning_rate = self.learning_rate * 1./np.sqrt(self.squared_grad_weights_accumulation + self.epsilon)
			biases_learning_rate = self.learning_rate * 1./np.sqrt(self.squared_grad_biases_accumulation + self.epsilon)

			# perform a gradient descent step using adaptative learning rates
			self.weights = self.weights - np.multiply(weights_learning_rate, grad_weights)
			self.biases = self.biases - np.multiply(biases_learning_rate, grad_biases)

			# update of accumulations of squarred gradients using gamma (memory size parameter)
			self.squared_grad_weights_accumulation = self.gamma * self.squared_grad_weights_accumulation + (1. - self.gamma) * grad_weights**2
			self.squared_grad_biases_accumulation = self.gamma * self.squared_grad_biases_accumulation + (1. - self.gamma) * grad_biases**2

		elif self.optim_method_name == "ADAM":
			# compute gradients for weights and biases parameters
			grad_weights = input.T.dot(grad_output) + grad_reg
			grad_biases = np.sum(grad_output, axis=0)

			# compute unbiased moments of order 1 and 2
			weights_moment_1 = self.grad_weights_accumulation / (1. - self.gamma_momentum)
			weights_moment_2 = self.squared_grad_weights_accumulation / (1. - self.gamma_adaptative)
			biases_moment_1 = self.grad_biases_accumulation / (1. - self.gamma_momentum)
			biases_moment_2 = self.squared_grad_biases_accumulation / (1. - self.gamma_adaptative)

			# compute adaptative learning rates using gradients accumulation
			weights_learning_rate = self.learning_rate * 1./(np.sqrt(weights_moment_2) + self.epsilon)
			biases_learning_rate = self.learning_rate * 1./(np.sqrt(biases_moment_2) + self.epsilon)

			# perform a gradient descent step using adaptative learning rates and momentum
			self.weights = self.weights - np.multiply(weights_learning_rate, weights_moment_1)
			self.biases = self.biases - np.multiply(biases_learning_rate, biases_moment_1)

			# update of accumulations of gradients and squarred gradients using gammas (memory size parameters)
			self.grad_weights_accumulation = self.gamma_momentum * self.grad_weights_accumulation + (1. - self.gamma_momentum) * grad_weights
			self.grad_biases_accumulation = self.gamma_momentum * self.grad_biases_accumulation + (1. - self.gamma_momentum) * grad_biases
			self.squared_grad_weights_accumulation = self.gamma_adaptative * self.squared_grad_weights_accumulation + (1. - self.gamma_adaptative) * grad_weights**2
			self.squared_grad_biases_accumulation = self.gamma_adaptative * self.squared_grad_biases_accumulation + (1. - self.gamma_adaptative) * grad_biases**2


		return grad_input

