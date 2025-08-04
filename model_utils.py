import torch 
import torch.nn as nn
import os
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Usage of LSTM long short term memory which is a type of recurrent neural network (RNN) architecture
# usfull for stock price prediction and language modeling tasks

class stockLSTM(nn.Module):
    def __init__(self,input_size, hidden_size=100, num_layers=3):
        super(stockLSTM, self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.2) # LSTM layer with input size, hidden size and number of layers with dropout of 0.2
        """
        LSTM parameters:
        Parameter    Meaning
        input_size   Number of input features per time step (e.g., Close, RSI, etc.)
        hidden_size  Number of neurons inside the LSTM cells (controls model capacity)
        num_layers   Stack multiple LSTM layers on top of each other
        batch_first  Ensures input is shaped as (batch, time_steps, features)
        dropout=0.2  Randomly drops 20% of neurons during training to prevent overfitting
        """

        self.fc=nn.Linear(hidden_size,1) # Fully connected layer to map the LSTM output to a single value (predicted stock price)
        # The input of the fully connected layer is the hidden size of the LSTM and the output is the predicted stock price of the next day

        def forward(self,x):
            """
            This part tell us how the data flows through the model.
            how to pass the input data through the LSTM layer and then through the fully connected layer.
            """
            out=self.lstm(x) # Pass the input through the LSTM layer
            out=out[:,-1,:] #Get the output of the last time step(the last day) from the LSTM output
            return self.fc(out) # Pass the last time step output through the fully connected layer to get the predicted stock price
        
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients before the backward pass
            out = model(seq)       # Forward pass through the model
            loss = criterion(out, labels)  # Calculate the loss
            loss.backward()        # Backward pass to compute gradients
            optimizer.step()       # Update the model parameters using the optimizer
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    print("Training complete.")

def evaluate_model(model, X_test, y_test):
    """
    This function evaluates a trained PyTorch model on the test dataset.
    It runs the model on the test data, compares predicted values vs actual values,
    and computes error metrics:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
    Returns these metrics to indicate model performance on unseen data.
    """
    model.eval() # set the model to evalutaion mode
    """
    When the model is in evalutaion mode it will not update the weights and biases of the model.
    This means that the model will turn off the dropout layers and batch normalization layer updates
    """

    with torch.no_grad(): #disable gradident calculation
        """
        we are not tracking the model gradients we will disable it to save memory and computation time"""
        y_pred=model(X_test).squeeze().numpy() # gives the predicted stock in numpy array format
        """
        X_test is used to forward pass through the model to get the predicted stock price.
        squeeze() removes the dimensions of size 1 from the tensor
        this is useful when we want to convert the tensor to a numpy array
        rsme and rae are used to work with the numpy array format
        """
        y_true=y_test.squeeze().numpy() # gives the actual stock price in numpy array format
        """
        same squeezing and converstion to numpy array as above
        """
    rsme=np.sqrt(mean_squared_error(y_true,y_pred))
    mae=mean_absolute_error(y_true,y_pred)
    return rsme, mae

def save_model(model,model_path):
    """
    Saves the trained PyTorch model to a specified file path.
    This allows for later loading and inference without retraining.
    """
    torch.save(model.state_dict(), model_path)  # Save only the model parameters (state_dict)
    print(f"Model saved to {model_path}")   

def load_model(model, model_path):
    """
    Loads a saved PyTorch model from a specified file path.
    This allows for reusing the trained model without retraining.
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))  # Load the model parameters
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")    
        """ If the model file does not exist, raise an error.
        This ensures that the user is aware of the missing model file.
        """         
    return model  # Return the model with loaded parameters

def predict_future(model,last_seq,n_days=7):
    """
    Predicts future stock prices for a specified number of days using the trained model.
    predicts the next 7 days of stock prices based on the last available sequence here its the last 60 days of data.
    Uses its own predictions as input for subsequent days.
    """
    model.eval() # Set the model to evaluation mode
    preds=[] # List to store predictions
    seq=last_seq.clone() # Clone the last sequence to avoid modifying the original data , ensures the original sequence remains unchanged.
    """
    The method which is used here is called as predictive forcasting.
    It uses the last sequence of data to predict the next day stock price.
    """
    for i in range(n_days): # Loop for the number of days to predict
        with torch.no_grad(): # Disable gradient calculation for efficiency
            pred=model(seq.unsqueeze(0)).item() # Forward pass to get the predicted stock price for the next day
            """
            unsqueeze(0) adds a batch dimension to the sequence as the model expects input in the shape of (batch_size, time_steps, features)
            item() converts the tensor to a Python number (float) for easier handling.
            """
        preds.append(pred) # Append the prediction to the list.
        new_input=torch.tensor([[pred]*seq.shape[1]],dtype=torch.float32) # Create a new input tensor for the next prediction
        """
        seq.shape[1] gives the number of features in the sequence (e.g., Close, RSI, etc.)
        [pred]*seq.shape[1] creates a new input tensor with the predicted value repeated for each feature in the sequence.
        This is a univariate time series prediction where we only predict the next stock price based on the previous price and also the close price.
        The usage of rsi, ema, sma etc is not used here and only used for understanding what are the technical indicators that are used in the stock market.
        """
        seq=torch.cat((seq[1:],new_input),dim=0) #Updating the sequence by removing the oldest value and adding the new predicted value.
        """
        The first row oldest value is dropped and the new predicted value is added to the end of the sequence.
        This ensures that the sequence always contains the most recent data for the next prediction.
        """
    return preds # List of predicted stock prices for the next n_days.




