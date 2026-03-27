import numpy as np

def create_sequences(data, seq_length):
    """
    Creates sequences of length seq_length from the data.
    X represents the past `seq_length` steps, and y represents the target feature at `seq_length` step.
    Assuming the target feature to predict is the last column.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, -1])
        
    return np.array(X), np.array(y)