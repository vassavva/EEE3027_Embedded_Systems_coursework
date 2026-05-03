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

debug = True


#Loading Class

class Load(threading.Thread):

    def __init__(self,  debug: bool):
        super(Load, self).__init__()
        self.debug = debug
    
        
    def run(self):

        # Load the weather data
        weather = pd.read_csv(r'C:\Users\vassa\OneDrive\Έγγραφα\EEE3027\labs\coursework\weather.csv') #add directory of file
        
        # Convert the date from integer to a datetime object
        weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')
        weather['snow_depth'] = weather['snow_depth'].fillna(0)
        weather.dropna(inplace=True) 
        
     
        months = weather['date'].dt.month
        weather['month_sin'] = np.sin(2 * np.pi * months / 12)
        weather['month_cos'] = np.cos(2 * np.pi * months / 12)

        # Generate the lagged data - Suppose for each time step, we want to consider the previous day
        lagged_weather = weather.copy()
        for c in lagged_weather.columns:
            lagged_weather[c] = lagged_weather[c].shift(1)
        lagged_weather.dropna(inplace = True) # Because of lag shifts, you should delete the rows with the introduced NaNs

        # Remove first element from weather as it is no longer applicable
        weather_ahead = weather.copy()
        weather_ahead = weather_ahead.iloc[1:]

       

        # Only make this true if you want to see this print statement
        if debug:
            print("Original data")
            print(weather)
            print("Lag 1 data")
            print(lagged_weather)
            print("Weather ahead 1 data")
            print(weather_ahead)

        self.weather_ahead = weather_ahead
        self.lagged_weather = lagged_weather
        

            
#Split class into training and testing sets
class Split(threading.Thread):
    def __init__(self,lagged_weather,weather_ahead,debug:bool):    
        super(Split, self).__init__()
        self.lagged_weather =lagged_weather
        self.weather_ahead = weather_ahead
        self.debug = debug
        


        
    def run(self):
        input_features = ['min_temp', 'max_temp', 'cloud_cover', 'sunshine', 'global_radiation', 'pressure']
        input_data = self.lagged_weather[input_features] 

        # ...while our target feature will refer to weather to avoid potential data leakage
        target_feature = ['mean_temp']
        target_data = self.weather_ahead[target_feature] 

        if debug:
            print("Input data")
            print(input_data)
            print("Target data")
            print(target_data)

        # Flatten the data to be in NumPY format
        input_data = input_data.values
        target_data = target_data.values

        # Split the data
        training_split = 0.7
        max_training_index = int(len(input_data) * training_split)

        # X becomes our input and y becomes our output/target
        X_train, X_test = input_data[:max_training_index], input_data[max_training_index:] # Note the positions of : to indicate the first 70% and last 30%
        y_train, y_test = target_data[:max_training_index], target_data[max_training_index:]

        
        # Debug print statements to show the array dimensions and the contents
        if debug:
            print(f"X_train ({X_train.shape}): {X_train}")
            print(f"X_test ({X_test.shape}): {X_test}")
            print(f"y_train ({y_train.shape}): {y_train}")
            print(f"y_test ({y_test.shape}): {y_test}")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.input_features = input_features
        self.target_feature = target_feature

#Preprocess class
class Preprocess(threading.Thread):
    def __init__(self,X, y, debug:bool):
        super(Preprocess, self).__init__()
        self.X = X
        self.y = y
        self.debug = debug
        
    def run(self):
        # Scale the data to *only* the training data to avoid data leakage
        feature_scaler = StandardScaler()
        feature_scaler.fit(self.X) 
        scaled_X = feature_scaler.transform(self.X)

        target_scaler = StandardScaler()
        target_scaler.fit(self.y.reshape(-1, 1))
        scaled_y= target_scaler.transform(self.y.reshape(-1, 1))

        if debug:
            print(f"scaled_X_[{self.name}]({scaled_X.shape}): {scaled_X}")
            print(f"scaled_y_[{self.name} ({scaled_y.shape}): {scaled_y}")
        
        self.scaled_X = scaled_X
        self.scaled_y = scaled_y

       
    def generate_frames_from_data_sources(
        input_data        : np.array,
        target_data       : np.array,
        input_data_length : int, 
        forecast_horizon  : int
    ) -> np.array:
        """
        Generate an np.array of 'frames' for the input and target data sources

        data (np.array) is the data source you which to 'frame' the data from.
        input_data_length (int) is how many 'days' you want to consider as your input
        forecast_horizon (int) is how many 'days' ahead you want to predict, e.g. given f(i_{t-input_len, ..., t}) = o_{t+1, ..., t+forecast_horizon}
        """
        assert len(input_data) == len(target_data), f"Mismatch of input data length and target data length. Input shape: {input_data.shape}, Target shape: {target_data.shape} "
        considered_length = len(input_data) - (input_data_length + forecast_horizon) # Generate frames only between the valid periods
        input_frames  = []
        target_frames = []
        for i in range(considered_length): 
            curr_input_max_idx = i + input_data_length
            curr_target_max_idx = curr_input_max_idx + forecast_horizon
            input_frames.append(input_data[i : curr_input_max_idx])
            target_frames.append(target_data[curr_input_max_idx : curr_target_max_idx])
        
        return np.array(input_frames), np.array(target_frames)
        # Debug print statements to show the array dimensions and the contents
       
        



#Batching class
class Batch(threading.Thread):
    def __init__(self,input_days,forecast_horizon, batch :int, X,y, debug:bool ):
        super(Batch,self).__init__()
        self.input_days = input_days
        self.forecast_horizon = forecast_horizon
        self.X = X
        self.y= y
        self.batch = batch
        self.debug = debug
        

    def run(self):
        

        X_frames, y_frames = Preprocess.generate_frames_from_data_sources(self.X, self.y, self.input_days, self.forecast_horizon)


        if debug:
            print(f"X_frames ({X_frames.shape}): {X_frames}")
            print(f"y_frames ({y_frames.shape}): {y_frames}")
            

        X_frames_t = torch.tensor(X_frames, dtype=torch.float32)
        y_frames_t = torch.tensor(y_frames, dtype=torch.float32)
        dataset = TensorDataset(X_frames_t, y_frames_t)

        

        loader = DataLoader(dataset, batch_size = self.batch ,shuffle= True,drop_last  = False)
        self.dataset = dataset
        self.loader = loader
        


#LSTM Wrapper class
class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMWrapper, self).__init__()
        self.M = hidden_dim
        self.L = layer_dim
        
        ## The actual underlying LSTM model
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        # initial hidden state and cell state
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        
        out, (hn, cn) = self.model(X, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out



#training class
# MSE can be a bit much for timeseries. RMSE may perform better.
class RMSELoss(torch.nn.Module):
    """
    Borrowing function from: https://discuss.pytorch.org/t/rmse-loss-function/16540
    """
    def __init__(self, eps = 1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, predicted, actual):
        return torch.sqrt(self.mse(predicted, actual) + self.eps)


class Train(threading.Thread):
    def __init__(self,
        wrapper : LSTMWrapper,
        train_loader : DataLoader, 
        test_loader : DataLoader, 
        epochs : int,
        learning_rate : float = 0.01
    ):
        super(Train,self).__init__()
        self.wrapper = wrapper
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def run(self):
        # criterion = nn.MSELoss() ### Use one or the other. Not both
        criterion = RMSELoss()
        optimizer = torch.optim.Adam(self.wrapper.parameters(), lr = self.learning_rate)

        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):

            # --- TRAINING ---
            self.wrapper.train() # Set model for training
            batch_train_losses = []
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.wrapper(batch_X)
                loss = criterion(outputs, batch_y.squeeze(-1))
                loss.backward()
                optimizer.step()
                batch_train_losses.append(loss.item())
            train_losses.append(np.mean(batch_train_losses)) # Average train loss for the epoch

            # --- EVALUATION ---
            self.wrapper.eval() # Set model for 'inference'
            batch_test_losses = []
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.wrapper(batch_X)
                    loss = criterion(outputs, batch_y.squeeze(-1))
                    batch_test_losses.append(loss.item())
            test_losses.append(np.mean(batch_test_losses)) # Average test loss for the epoch

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1} | Train RMSE: {train_losses[-1]:.4f} | Test RMSE: {test_losses[-1]:.4f}')

            self.test_losses = test_losses
            self.train_losses= train_losses


   


#Infer Class

class Infer(threading.Thread):
    def __init__(self, test_dataset , how_many_samples_to_view, lstm_wrapper):
        super(Infer , self ).__init__()
        self.test_dataset = test_dataset
        self.how_many_samples_to_view = how_many_samples_to_view
        self.lstm_wrapper = lstm_wrapper


    def run(self):
        test_samples_to_observe = [random.randint(0, len(self.test_dataset) - 1) for _ in range(self.how_many_samples_to_view)]
        for i in test_samples_to_observe:
            X_in, y_out = self.test_dataset[i]
            X_in_t = X_in.reshape(1, X_in.shape[0], X_in.shape[1]).to(device)
            outputs = self.lstm_wrapper(X_in_t) # Create an auxiliary 'batch' to feed in the tensor to the ML model
            outputs = outputs.reshape(outputs.shape[1], outputs.shape[0])
            Plot.plot_framed_data(
                self.target_scaler.inverse_transform(outputs.detach().cpu().numpy()), 
                self.target_scaler.inverse_transform(y_out.detach().cpu().numpy())
            )


#plot class
class Plot(threading.Thread):
    def __init__(self,train_losses , test_losses,how_many_samples_to_view, test_dataset):
        super(Plot,self).__init__()
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.how_many_samples_to_view = how_many_samples_to_view

    def run(self):

        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.test_losses, label='test loss')
        plt.xlabel('epoch no')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
 
 
    def plot_framed_data(
        model_prediction, 
        target_data,
        figsize = (12, 8)
    ):
        plt.figure(figsize = figsize)
        plt.plot(model_prediction, label = "LSTM Output", linewidth = 1)
        plt.plot(target_data, label = "Target Output", linewidth = 1)
        plt.xlabel('No. of Days')
        plt.ylabel('Mean Temperature (Celcius)')
        plt.legend(loc='lower right')


#analysing class
class Analyse(threading.Thread):
    def __init__(self, test_dataset ,lstm_wrapper):
        super(Analyse, self).__init__()
        self.test_dataset = test_dataset
        self.lstm_wrapper = lstm_wrapper

    def run(self):

        full_model_prediction = []
        expected_output = []
        for i in range(len(self.test_dataset)):
            X_in, y_out = self.test_dataset[i]
            X_in_t = X_in.reshape(1, X_in.shape[0], X_in.shape[1]).to(device)
            outputs = self.lstm_wrapper(X_in_t) # Create an auxiliary 'batch' to feed in the tensor to the ML model
            outputs = outputs.reshape(outputs.shape[1], outputs.shape[0])
            full_model_prediction.append(outputs[0, 0].detach().cpu().numpy())
            expected_output.append(y_out[0, 0].detach().cpu().numpy())