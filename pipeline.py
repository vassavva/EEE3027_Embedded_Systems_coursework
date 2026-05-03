import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import random 
import time
import threading

from sklearn.preprocessing import StandardScaler
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from Part_1 import  Buffer
from Part_2 import Load, Split , Preprocess, Batch ,LSTMWrapper , Train , Infer , Plot ,Analyse



warnings.filterwarnings('ignore')

#Setting necessary device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device) # Debug printing

# Setting necessary seeds for reproducibility
RANDOM_SEED = 166
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


#variables that are common for multiple classes
forecast_horizon = 7
input_days= 30
view  = 5 #how_many_samples_to_view variable for Infer and Plot classes

#pipeline

#data_buffer = Buffer(size=10)
train_buffer = Buffer(size =  10 )
test_buffer = Buffer(size=  10  )
test_dataset_buffer = Buffer(size =  10)
losses_buffer = Buffer(size= 10 )
wrapper_buffer = Buffer(size = 10)

#start time to calculate the time taken for the pipeline to run 
start_time = time.time() 

# Step 1:load Data
print('Start Loading\n')
load = Load(debug=True)

load.start()
load.join()

#Step 2: Split Data
print('Start Spliting\n')
split = Split(weather_ahead = load.weather_ahead, lagged_weather = load.lagged_weather , debug = True)

split.start()
split.join()

print('Adding split data into the two buffers\n')
train_buffer.produce((split.X_train, split.y_train))
test_buffer.produce((split.X_test , split.y_test))

#train and test run concurrently 
#Step 3 :Preprocess
print('Data are retrieved from the buffers\n')
X_train, y_train = train_buffer.consume()
X_test,  y_test  = test_buffer.consume()

print('Start Preprocessing for train and test data concurently\n') 
preprocess_train = Preprocess(X=X_train, y=y_train, debug=True)
preprocess_test  = Preprocess(X=X_test,  y=y_test,  debug=True)


preprocess_train.start()
preprocess_test.start()
preprocess_train.join()
preprocess_test.join()

#Step 4: Batching
print('Start Batching for train and test data concurrently\n')
batch_train = Batch (input_days = input_days , forecast_horizon = forecast_horizon, batch = 32, X = preprocess_train.scaled_X ,y = preprocess_train.scaled_y, debug = True)
batch_test = Batch (input_days = input_days , forecast_horizon= forecast_horizon,batch=10 , X = preprocess_test.scaled_X , y = preprocess_test.scaled_y, debug= True)

batch_train.start()
batch_test.start()
batch_train.join()
batch_test.join()

#add output into buffers
print('loaders and dataset are added into the buffer\n')
train_buffer.produce(batch_train.loader)
test_buffer.produce(batch_test.loader)
test_dataset_buffer.produce(batch_test.dataset)


#calculations for LSTM wrapepr
print('Initialising the LSTM Wrapper\n')
no_of_outputs    = 1
no_lstm_outputs  = forecast_horizon * no_of_outputs  
no_of_inputs = len(split.input_features)                     # This needs to be the number of features per timestep
no_of_outputs = len(split.target_feature)                    # number of target variables (per timestep)
no_of_lstm_outputs = forecast_horizon * no_of_outputs  # and multiple it by how many timesteps ahead we want


#Initialise LSTM WRAPPER for Train Step
lstm_wrapper = LSTMWrapper(input_dim = no_of_inputs , hidden_dim= 128 , layer_dim = 2 , output_dim = no_lstm_outputs).to(device)


#gets loaders from buffer for train class
print('Retrieving loaders from buffers\n')
train_loader = train_buffer.consume()
test_loader = test_buffer.consume()

#Step 5:Training
print('Starts Training\n')
train = Train(wrapper = lstm_wrapper , train_loader = train_loader ,test_loader=test_loader, epochs =200 , learning_rate = 0.01)

train.start()
train.join()

#adds losses into buffer
print('adding losses into buffers\n')
losses = losses_buffer.produce((train.test_losses , train.train_losses))

print('retreiving dataset and losses from buffers\n')
dataset = test_dataset_buffer.consume()
train , test = losses_buffer.consume()

print('Start Inference, Plot and Analyse concurrently')
#infer  = Infer(test_dataset= dataset, how_many_samples_to_view= view , lstm_wrapper= lstm_wrapper )
#plot = Plot(train_losses= train  , test_losses= test  , how_many_samples_to_view=  view , test_dataset = dataset)
analyse = Analyse(test_dataset = dataset , lstm_wrapper=lstm_wrapper )

analyse.start()
#infer.start()
#plot.start()


analyse.join()
#infer.join()
#plot.join()

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")