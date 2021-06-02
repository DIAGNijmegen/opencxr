from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization, Flatten
from keras.initializers import he_normal
from keras.optimizers import Adam, SGD
from keras.initializers import he_normal

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception

def resnet_notop(inputs):
    inputs = Input((256,256,3))
    resnet = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    x = resnet.output
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    
    return x


def build_multi_output1(input_shape = (256, 256, 3)):
    inputs = Input(input_shape)
    resnet = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    x = resnet.output
    x = Flatten()(x)

    z1 = Dropout(0.25)(x)
    y1 = Dense(2, activation='softmax', name="type")(z1)

    z2 = Dropout(0.25)(x)
    y2 = Dense(4, activation='softmax', name="rot")(z2)

    model = Model(inputs=inputs, outputs=[y1, y2])

    #optimizer = Adam(lr=0.001)
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    losses = {"type": "categorical_crossentropy",
	            "rot": "categorical_crossentropy"}
    lossWeights = {"type": 1.0, "rot": 1.0}
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, metrics=metrics)

    return model


def build_multi_output2(input_shape = (256, 256, 3)):
    inputs = Input(input_shape)
    resnet = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    x = resnet.output
    x = Flatten()(x)

    z1 = Dropout(0.25)(x)
    y1 = Dense(2, activation='softmax', name="type")(z1)

    z2 = Dropout(0.25)(x)
    y2 = Dense(4, activation='softmax', name="rot")(z2)

    z3 = Dropout(0.25)(x)
    y3 = Dense(1, activation='sigmoid', name="inv")(z3)

    model = Model(inputs=inputs, outputs=[y1, y2, y3])

    #optimizer = Adam(lr=0.001)
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    losses = {"type": "categorical_crossentropy",
	            "rot": "categorical_crossentropy",
                "inv": "binary_crossentropy"}
    lossWeights = {"type": 1.0, "rot": 1.0, "inv":0.33}
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, metrics=metrics)
    return model


def build_multi_output3(input_shape = (256, 256, 3)):
    inputs = Input(input_shape)
    resnet = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    x = resnet.output
    x = Flatten()(x)

    z1 = Dropout(0.5)(x)
    z1 = Dense(128, activation='relu', kernel_initializer='he_normal')(z1)
    y1 = Dense(4, activation='softmax', name="type")(z1)

    z2 = Dropout(0.5)(x)
    z2 = Dense(128, activation='relu', kernel_initializer='he_normal')(z2)
    y2 = Dense(4, activation='softmax', name="rot")(z2)
    

    z3 = Dropout(0.5)(x)
    z3 = Dense(128, activation='relu', kernel_initializer='he_normal')(z3)
    y3 = Dense(1, activation='sigmoid', name="inv")(z3)

    z4 = Dropout(0.5)(x)
    z4 = Dense(128, activation='relu', kernel_initializer='he_normal')(z4)
    y4 = Dense(1, activation='sigmoid', name="flip")(z4)

    model = Model(inputs=inputs, outputs=[y1, y2, y3, y4])

    #optimizer = Adam(lr=0.001)
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    losses = {"type": "categorical_crossentropy",
	            "rot": "categorical_crossentropy",
                "inv": "binary_crossentropy",
                "flip": "binary_crossentropy"}
    lossWeights = {"type": 1.0, "rot": 1.0, "inv":1.0, "flip":1.0}
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, metrics=metrics)

    return model
