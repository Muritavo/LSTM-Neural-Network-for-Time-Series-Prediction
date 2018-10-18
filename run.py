__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import sys
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

model = None

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    offset = len(true_data) - len(predicted_data) * prediction_len
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len + offset)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def main():
	configs = json.load(open(CONFIG, 'r'))

	data = DataLoader(
		DATA,
		configs['data']['train_test_split'],
		configs['data']['columns']
	)

	model = Model()
	model.build_model(configs)
	x, y = data.get_train_data(
		seq_len = configs['data']['sequence_length'],
		normalise = configs['data']['normalise']
	)

	'''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size']
	)
	'''
	# out-of memory generative training
	steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
	model.train_generator(
		data_gen = data.generate_train_batch(
			seq_len = configs['data']['sequence_length'],
			batch_size = configs['training']['batch_size'],
			normalise = configs['data']['normalise']
		),
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		steps_per_epoch = steps_per_epoch,
		model_path = MODEL
	)
	
	x_test, y_test = data.get_test_data(
		seq_len = configs['data']['sequence_length'],
		normalise = configs['data']['normalise']
	)

	predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
	#predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
	#predictions = model.predict_point_by_point(x_test)        

	plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
	#plot_results(predictions, y_test)
	sys.stdout.write("--END--")

def predict():
	configs = json.load(open(CONFIG, 'r'))

	data = DataLoader(
		DATA,
		configs['data']['train_test_split'],
		configs['data']['columns']
	)
	
	global model
	if model == None:
		model = Model()
		model.load_model(MODEL)
	
	x_test, y_test = data.get_test_data(
		seq_len = configs['data']['sequence_length'],
		normalise = configs['data']['normalise']
	)

	if TYPE == "sequence":
		predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
		plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
	if TYPE == "point" or TYPE == "predict":
		predictions = model.predict_point_by_point(x_test)
	if TYPE == "full":
		predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
	if TYPE == "full" or TYPE == "point":
		plot_results(predictions, y_test)
	if TYPE == "predict":
		predicted_value = data.denormalize_windows(predictions[-1], configs['data']['sequence_length'])
		sys.stdout.write("--END--{}--END--\n".format(predicted_value))
	else:
		sys.stdout.write("--END--")

		
while True:
	print("Trying")
	ARGS = input()
	if ARGS != None:
		ARGS = ARGS.split(";")
		print(ARGS)
		CONFIG = ARGS[0]
		MODEL = ARGS[1]
		DATA = ARGS[2]
		OPERATION = ARGS[3]
		TYPE = ARGS[4]
		if OPERATION == "train":
			main()
		else:
			predict()
	time.sleep(1)