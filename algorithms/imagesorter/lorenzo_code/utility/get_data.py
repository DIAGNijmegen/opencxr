import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def get_data_list1(data_csv_path, labels=["Frontal", "Lateral"], split=None, strat=False):
    # Get list of images(path) and corresponding labels from the csv file
    # if split -> train-validation split (e.g. split = 0.2)
    
    df = pd.read_csv(data_csv_path)
    
    x = df["Image_path"].values
    
    y = np.empty((len(x),len(labels)), dtype=int)

    for i in range(len(labels)):
        y[:,i] = df[labels[i]].values
    
    if split is not None:
        if strat :
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split, stratify=y[:,0])
        else:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split)
        
        return x_train, x_val, y_train, y_val
    else:
        return x, y

def get_data_list2(data_csv_path, labels=["img_type", "rotation", "inversion", "lateral_flip"], split=None, strat=False, seed=None):
    # Get list of images(path) and corresponding labels from the csv file
    # if split -> train-validation split (e.g. split = 0.2)
    
    df = pd.read_csv(data_csv_path)
    
    x = df["new_img_name"].values
    
    y = np.empty((len(x),len(labels)), dtype=int)

    for i in range(len(labels)):
        y[:,i] = df[labels[i]].values
    
    if split is not None:
        if strat :
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split, stratify=y[:,0], random_state=seed)
        else:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split, random_state=seed)
        
        return x_train, x_val, y_train, y_val
    else:
        return x, y


def get_y(labels):

    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis = 0)

    out = np.zeros((len(labels), 8), dtype=int)

    for i in range(len(labels)):
        if labels[i, 0] == 0:
            out[i, 0] = 1
            
        if labels[i, 0] == 1:
            out[i, 1] = 1

        if labels[i, 1] == 0:
            out[i, 2] = 1
        if labels[i, 1] == 1:
            out[i, 3] = 1
        if labels[i, 1] == 2:
            out[i, 4] = 1
        if labels[i, 1] == 3:
            out[i, 5] = 1
        
        if labels[i, 2] == 1:
            out[i, 6] = 1
            
        if labels[i, 3] == 1:
            out[i, 7] = 1

    return out

