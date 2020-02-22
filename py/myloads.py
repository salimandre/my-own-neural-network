# required Libraries
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from itertools import islice
import random
import time

# required .py Files
from layer import Layer
from relu import ReLU
from dense import Dense
from loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from back_and_forth import forward, predict, train 
from generate_data import get_data, iterate_minibatches, show_data