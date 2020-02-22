from myloads import *

def get_data():
	""" Load/Download and preprocess mnist datasets (flattening, rescaling) """
	(X_train, y_train), (X_val, y_val) = mnist.load_data()
	n_features = X_train.shape[1]*X_train.shape[1]
	n_train = X_train.shape[0]
	n_val = X_val.shape[0]
	X_train = X_train.reshape(n_train, n_features)
	X_val = X_val.reshape(n_val, n_features)
	X_train = X_train.astype("float32") / 255
	X_val = X_val.astype("float32") / 255

	return (X_train, y_train), (X_val, y_val)

def iterate_minibatches(X_,y_,batchsize_,shuffle_=True):
	""" Randomly yield batches of X and y  """
	indx = [i for i in range(len(X_))]
	random.shuffle(indx)
	indxgenerator = (i for i in indx)
	del indx
	while True:
		batch_indx = list(islice(indxgenerator, batchsize_))
		if len(batch_indx)==0:
			break;
		else:
			yield X_[batch_indx,:], y_[batch_indx]

def show_data(X_train_, y_train_):
	plt.figure(figsize=[6,6])
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.title("Label: %i"%y_train_[i])
		plt.imshow(X_train_[i].reshape([28,28]),cmap='gray');
	plt.show()