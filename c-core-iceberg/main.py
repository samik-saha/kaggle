#%%
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#%%
x_train_orig, y_train = load_dataset()
print("x_train_orig shape: " + str(x_train_orig.shape))
print("y_train shpae: " + str(y_train.shape))
x_train=np.reshape(x_train_orig,(1605,75,75,1))
print("x_train shape: " + str(x_train.shape))

#%%
plt.imshow(x_train[600,:,:,0])
print(y_train[600])

#%%
def MyModel(input_shape):
    X_input = Input(input_shape)

    # X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(6, (7,7), strides = (1, 1), padding='same', name = 'conv0')(X_input)
    # X = BatchNormalization(axis=3, name = 'bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='MyModel')

    return model

#%%
model = MyModel((75,75,1))
model.reset_states()
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64)
