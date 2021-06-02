import numpy as np
import SimpleITK as sitk
import skimage.io as io
from PIL import Image
import keras
from keras.utils import to_categorical

class DataGenerator_1(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgs_list, labels, batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=(2,4), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_list = imgs_list
        self.list_IDs = np.arange(len(imgs_list))
        self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y1 = np.empty((self.batch_size, self.n_classes[0]))
        y2 = np.empty((self.batch_size, self.n_classes[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample (check for extension)
            #ext = str(self.imgs_list[ID].split('.')[-1])

            # this method convert a grayscale to rgb (8 bit)
            im_arr = io.imread(str(self.imgs_list[ID]), plugin='simpleitk')
            im = Image.fromarray((im_arr/256).astype(np.uint8), 'P')
            if im_arr.shape is not self.dim:
                im = im.resize(self.dim, resample=Image.BILINEAR)
            im = np.asarray(im.convert('RGB'), dtype=np.uint8)

            


            # Resize part
            #if im.shape is not self.dim:
                #dummy_spacing = (1, 1)
                #im, _ = utils_general.resize_to_x_y(im, dummy_spacing, self.dim[0], self.dim[1])
            
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            X[i,] = im/255.
            
            ####Data augmentation part if needed !!

            # Store target labels
            y1[i] = to_categorical(self.labels[ID][0], self.n_classes[0])  # type
            y2[i] = to_categorical(self.labels[ID][1], self.n_classes[1])  # rot

            # concatenate targets
            y = [y1, y2]

        return X, y


class DataGenerator_2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgs_list, labels, batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=(2,4,1), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_list = imgs_list
        self.list_IDs = np.arange(len(imgs_list))
        self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y1 = np.empty((self.batch_size, self.n_classes[0]))
        y2 = np.empty((self.batch_size, self.n_classes[1]))
        y3 = np.empty((self.batch_size, self.n_classes[2]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample (check for extension)
            #ext = str(self.imgs_list[ID].split('.')[-1])

            # this method convert a grayscale to rgb (8 bit)
            im_arr = io.imread(str(self.imgs_list[ID]), plugin='simpleitk')
            im = Image.fromarray((im_arr/256).astype(np.uint8), 'P')
            if im_arr.shape is not self.dim:
                im = im.resize(self.dim, resample=Image.BILINEAR)
            im = np.asarray(im.convert('RGB'), dtype=np.uint8)

            


            # Resize part
            #if im.shape is not self.dim:
                #dummy_spacing = (1, 1)
                #im, _ = utils_general.resize_to_x_y(im, dummy_spacing, self.dim[0], self.dim[1])
            
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            X[i,] = im/255.
            
            ####Data augmentation part if needed !!

            # Store target labels
            y1[i] = to_categorical(self.labels[ID][0], self.n_classes[0])  # type
            y2[i] = to_categorical(self.labels[ID][1], self.n_classes[1])  # rot
            y3[i] = self.labels[ID][2] # inv

            # concatenate targets
            y = [y1, y2, y3]

        return X, y

class DataGenerator_3(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgs_list, labels, batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=(4,4,1,1), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_list = imgs_list
        self.list_IDs = np.arange(len(imgs_list))
        #self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y1 = np.empty((self.batch_size, self.n_classes[0]))
        y2 = np.empty((self.batch_size, self.n_classes[1]))
        y3 = np.empty((self.batch_size, self.n_classes[2]))
        y4 = np.empty((self.batch_size, self.n_classes[3]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample (check for extension)
            #ext = str(self.imgs_list[ID].split('.')[-1])

            # this method convert a grayscale to rgb (8 bit)
            im_arr = io.imread(str(self.imgs_list[ID]), plugin='simpleitk')
            im = Image.fromarray((im_arr/256).astype(np.uint8), 'P')
            if im_arr.shape is not self.dim:
                im = im.resize(self.dim, resample=Image.BILINEAR)
            im = np.asarray(im.convert('RGB'), dtype=np.uint8)

            


            # Resize part
            #if im.shape is not self.dim:
                #dummy_spacing = (1, 1)
                #im, _ = utils_general.resize_to_x_y(im, dummy_spacing, self.dim[0], self.dim[1])
            
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            X[i,] = im/255.
            
            ####Data augmentation part if needed !!

            # Store target labels
            y1[i] = to_categorical(self.labels[ID][0], self.n_classes[0])  # type
            y2[i] = to_categorical(self.labels[ID][1], self.n_classes[1])  # rot
            y3[i] = self.labels[ID][2] # inv
            y4[i] = self.labels[ID][3] # flip

            # concatenate targets
            y = [y1, y2, y3, y4]

        return X, y


class DataGenerator_single_output(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgs_list, labels, batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_list = imgs_list
        self.list_IDs = np.arange(len(imgs_list))
        self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.n_classes))
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample (check for extension)
            #ext = str(self.imgs_list[ID].split('.')[-1])

            # this method convert a grayscale to rgb (8 bit)
            im_arr = io.imread(str(self.imgs_list[ID]), plugin='simpleitk')
            im = Image.fromarray((im_arr/256).astype(np.uint8), 'P')
            if im_arr.shape is not self.dim:
                im = im.resize(self.dim, resample=Image.BILINEAR)
            im = np.asarray(im.convert('RGB'), dtype=np.uint8)

            


            # Resize part
            #if im.shape is not self.dim:
                #dummy_spacing = (1, 1)
                #im, _ = utils_general.resize_to_x_y(im, dummy_spacing, self.dim[0], self.dim[1])
            
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            X[i,] = im/255.
            
            ####Data augmentation part if needed !!

            # Store target labels
            y[i] = self.labels[ID] 

        return X, y