from myloads import *

# return preprocessed train and val datasets
(X_train, y_train), (X_val, y_val) = get_data()

# show first images from train dataset
show_data(X_train,y_train)

# set architecture for the neural network 

initialization_option = "He&Zhang&Ren&Sun"
regularization_param = 0.0007
#optim_method_dict = {"name":"SGD", "learning_rate":0.1}
#optim_method_dict = {"name":"SGD_with_momentum", "learning_rate":0.1, "gamma_momentum":0.5}
#optim_method_dict = {"name":"NAG_method", "learning_rate":0.1, "gamma_momentum":0.1}
#optim_method_dict = {"name":"Adagrad", "learning_rate":0.1}
#optim_method_dict = {"name":"RMSProp", "learning_rate":0.001, "gamma":0.95}
optim_method_dict = {"name":"ADAM", "learning_rate":0.0008, "gamma_adaptative":0.98, "gamma_momentum":0.5}
print(optim_method_dict)

network = []
network.append(Dense(X_train.shape[1],100,init_option=initialization_option, reg_param=regularization_param, **optim_method_dict))#, learning_rate=learning_rate))
network.append(ReLU())
network.append(Dense(100,200,init_option=initialization_option, reg_param=regularization_param, **optim_method_dict))#, learning_rate=learning_rate))
network.append(ReLU())
network.append(Dense(200,10,init_option=initialization_option, reg_param=regularization_param, **optim_method_dict))#, learning_rate=learning_rate))

# training of the neural network

batchsize = 32
n_epochs = 25

train_log = []
val_log = []
for epoch in range(n_epochs):

	for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize_=batchsize,shuffle_=True):
		train(network,x_batch,y_batch)
	
	train_log.append(np.mean(predict(network,X_train)==y_train))
	val_log.append(np.mean(predict(network,X_val)==y_val))
	
	#clear_output()
	print("Epoch",epoch)
	print("Train accuracy:",train_log[-1])
	print("Val accuracy:",val_log[-1])

plt.plot(train_log,label='train accuracy')
plt.plot(val_log,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
