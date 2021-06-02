from __future__ import division
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import csv
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau, CSVLogger

from utility.get_data import get_data_list2
from utility.multi1 import build_multi_output3
from utility.generator import DataGenerator_3

def train(exp_n):

    csv_path = "/mnt/synology/cxr/projects/OpenCXR-imagesorter/processed_data/merged_data_augmented/new_train_frontal_split.csv"
    # train_dir_png = "/mnt/synology/cxr/projects/OpenCXR-imagesorter/processed_data/merged_data_augmented/train_png"
    train_dir_mha = "/mnt/synology/cxr/projects/OpenCXR-imagesorter/processed_data/merged_data_augmented/train_mha"

    # get x(images path) and y from csv and split dataset in train and val
    x_train, x_val, y_train, y_val = get_data_list2(csv_path, split= 0.2, strat=True, seed=92)

    # join path for images
    x_train = [os.path.join(train_dir_mha, img+'.mha') for img in x_train]
    x_val = [os.path.join(train_dir_mha, img+'.mha') for img in x_val]
    
    # create a folder for the experiment
    exp_dir = '/mnt/synology/cxr/projects/OpenCXR-imagesorter/lorenzo/exp'+str(exp_n)
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    print("OpenCXR Imagesorter - exp"+str(exp_n))

    # Training parameters
    batch_size = 32
    n_epoch = 100
    NUM_WORKERS = 16
    QUEUE_SIZE = 32

    # Parameters for generators
    train_params = {
        'dim': (256, 256),
        'batch_size': batch_size,
        'n_channels': 3,
        'n_classes': (4,4,1,1),
        'shuffle': True
    }

    val_params = {
        'dim': (256, 256),
        'batch_size': batch_size,
        'n_channels': 3,
        'n_classes': (4,4,1,1),
        'shuffle': True
    }
    
    model_path = os.path.join(exp_dir, 'image_sorter.hdf5')
    # Check if there is a save of the model to resum training
    if os.path.isfile(model_path):
        print('FOUND A SAVE OF tHE MODEL, GOING TO RESUME TRAINING')
        resnet = load_model(model_path)
    else:
        print("building model...")
        resnet = build_multi_output3(input_shape=(256,256,3))
        print("done.")

    # check if train can be resumed
    
    saved_log = os.path.join(exp_dir, 'train_log.csv')
    # check if the file exists or is empty (train stopped during the first epoch)
    if os.path.exists(saved_log) and os.stat(saved_log).st_size != 0:
        train_log = pd.read_csv(saved_log)

        # get current state of training
        current_state = train_log.index[train_log['val_loss'] == np.min(train_log['val_loss'].values)][0]
        EPOCH_INIT = current_state+1
        BEST_LOSS = train_log.loc[current_state]['val_loss']

        n_epoch = n_epoch - EPOCH_INIT #set target epoch's number to remaining epochs

        # delete epochs greater than current state
        train_log = train_log.loc[0:current_state]
        train_log.to_csv(saved_log, index=False)

        print('Resume training from EPOCH_INIT {0} ,BEST_LOSS {1}'.format(EPOCH_INIT+1, BEST_LOSS))
            
    else:
        BEST_LOSS = np.inf
        print('Start Training')

    #saved_log = os.path.join(exp_dir, 'train_log.csv')
    #if os.path.exists(saved_log):
    #    with open(saved_log) as f:
    #        reader = csv.DictReader(f)
    #        rows = [row for row in reader]
    #        EPOCH_INIT = int(rows[-1]['epoch'])+1
    #        BEST_LOSS = float(rows[-1]['val_loss'])
    #        LR = float(rows[-1]['lr'])
    #        print('Resume training from EPOCH_INIT {0} ,BEST_LOSS {1}, LR {2}'.format(EPOCH_INIT, BEST_LOSS, LR))
    #else:
    #    BEST_LOSS = np.inf
    
    cb_Early_Stop=EarlyStopping(monitor='val_loss',patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00001, cooldown=0, verbose=1)
    
    class ModelCheckpointWrapper(ModelCheckpoint):
        def __init__(self, best_init=None, *arg, **kwagrs):
            super(ModelCheckpointWrapper, self).__init__(*arg, **kwagrs)
            if best_init is not None:
                self.best = best_init

    checkpointer = ModelCheckpointWrapper(best_init=BEST_LOSS, filepath=model_path, verbose=1, save_best_only=True)
    #cb_Model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csvlogger = CSVLogger(saved_log, separator=',', append=True)    
    
    callbacks_list = [cb_Early_Stop, checkpointer, reduce_lr, csvlogger]

    # Generators
    training_generator = DataGenerator_3(x_train, y_train, **train_params)
    validation_generator = DataGenerator_3(x_val, y_val, **val_params)
            
    history = resnet.fit_generator(training_generator, validation_data=validation_generator,
                                    use_multiprocessing=True, workers=NUM_WORKERS, max_queue_size=QUEUE_SIZE,
                                     epochs=n_epoch, callbacks=callbacks_list, verbose=2)
    

    print("finish.")

    type_acc = history.history['val_type_acc']
    print("Type Accuracy is :")
    print(type_acc)
    print("Best Type accuracy is {} at epoch {}.".format(np.max(type_acc), np.where(type_acc == np.max(type_acc))[0][0]+1))
    print('#*40')

    rot_acc = history.history['val_rot_acc']
    print("Rot Accuracy is :")
    print(rot_acc)
    print("Best Rot accuracy is {} at epoch {}.".format(np.max(rot_acc), np.where(rot_acc == np.max(rot_acc))[0][0]+1))
    print('#*40')
    
    inv_acc = history.history['val_inv_acc']
    print("Inv Accuracy is :")
    print(inv_acc)
    print("Best Inv accuracy is {} at epoch {}.".format(np.max(inv_acc), np.where(inv_acc == np.max(inv_acc))[0][0]+1))
    print('#*40')
    
    flip_acc = history.history['val_flip_acc']
    print("Flip Accuracy is :")
    print(flip_acc)
    print("Best Flip accuracy is {} at epoch {}.".format(np.max(flip_acc), np.where(flip_acc == np.max(flip_acc))[0][0]+1))
    print('#*40')