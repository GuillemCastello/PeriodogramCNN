import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(gpus)
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.set_visible_devices(gpus[0], 'GPU')


def noise_model(x, a, alpha, b):
    return a*x**(-alpha) + b

frequency = np.linspace(1/(1440*60), 1/120, 2000).astype(np.float32)

with h5py.File('/home/guillem/synth_data.h5', 'r') as f:
    params = np.array(f['params'][:])
    #conf_params = f['conf_params'][:]
    pers = np.array(f['periodograms'][:])


re_pers = np.zeros((2000, pers.shape[1]*pers.shape[2]))
re_params = np.zeros((3, params.shape[0]*params.shape[1]))

for i in range(pers.shape[0]):
    for j in range(pers.shape[1]):
        re_pers[:, i*pers.shape[1]+j] = pers[i, j, :]
        re_params[:, i*pers.shape[1]+j] = params[i, j, :]

zeros = []
for id, param in enumerate(re_params.T):
    if any(param <= 0):
        zeros.append(id)
    if any(param < 1e-15):
        zeros.append(id)

bads = []
for id, per in enumerate(re_pers.T):
    if np.max(per) > 1:
        bads.append(id)

delete = list(set(zeros + bads))
re_params_corr = np.delete(re_params, delete, axis=1)
re_pers_corr = np.delete(re_pers, delete, axis=1)

re_params_corr = re_params_corr.T
re_pers_corr = re_pers_corr.T

re_params_corr = np.log10(re_params_corr)


scaler_params = MinMaxScaler()
scaler_params.fit(re_params_corr)
re_params_corr = scaler_params.transform(re_params_corr)

params_train, params_val, \
per_train, per_val = train_test_split(re_params_corr,
                                    re_pers_corr, 
                                    test_size=0.2)


class CNN(Model):
    def __init__(self):
        """
        Initializes the CNN model.

        The model consists of a series of convolutional layers followed by average pooling, batch normalization,
        dropout, and dense layers. The output layer uses sigmoid activation for multi-label classification.

        Args:
            None

        Returns:
            None
        """
        super(CNN, self).__init__()
        self.original_dim = 2000
        self.dropout_rate = 0.2

        self.conv_block = tf.keras.Sequential([
            layers.Input(shape=(self.original_dim, 1)),

            layers.Conv1D(64, 5, activation='relu', padding='valid'),
            layers.Conv1D(64, 8, activation='relu', padding='valid'),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation='relu', padding='valid'),
            layers.Conv1D(64, 10, activation='relu', padding='valid'),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation='relu', padding='valid'),
            layers.Conv1D(64, 10, activation='relu', padding='valid'),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation='relu', padding='valid'),
            layers.Conv1D(64, 10, activation='relu', padding='valid'),
            layers.MaxPooling1D(2),

            layers.Flatten(),

            layers.Dense(1024, activation='relu'),
            layers.Dropout(self.dropout_rate),

            layers.Dense(512, activation='relu'),
            layers.Dropout(self.dropout_rate),

            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(3, activation='sigmoid')
        ])
        
    def call(self, input_features):
        """
        Executes a forward pass through the CNN model.

        Args:
            input_features (tf.Tensor): Input features to the model.

        Returns:
            tf.Tensor: Output tensor from the model.
        """
        x = tf.expand_dims(input_features, -1)
        x = self.conv_block(x)
        return x
    

optimizer_object = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.MeanSquaredError()

def scheduler(epoch, lr):
    if epoch >= 5 and epoch % 10 == 0:
        lr *= 0.8
    return lr

lr_sch = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Create the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitors the validation loss
    patience=7,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Verbosity mode
    mode='min',          # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

model = CNN()
model.build(input_shape=(None, 2000))
model.compile(optimizer=optimizer_object, loss=loss_object)

history = model.fit(per_train, params_train, validation_data=(per_val, params_val), batch_size=32, epochs=100, callbacks=[lr_sch, early_stopping])
weights_file_name = 'BestFitWeights_NEW.h5'
model.save_weights(f'./CNN/BestFit/{weights_file_name}')
model.save('./CNN/BestFit/BestFitModel_NEW')