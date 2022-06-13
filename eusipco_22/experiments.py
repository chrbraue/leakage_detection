import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

print(tf.test.is_gpu_available())

# seeds to make results reproducible
np.random.seed(0)
tf.random.set_seed(0)

# auxiliary function to rotate samples
def rotate90(df):
    df_90 = df.copy()
    df_90['mfc1'] = df['mfc4']
    df_90['mfc2'] = df['mfc1']
    df_90['mfc3'] = df['mfc2']
    df_90['mfc4'] = df['mfc3']
    for i in [1, 2, 3]:
        df_90[f'x{i}'] = 1 - df[f'y{i}']
        df_90[f'y{i}'] = df[f'x{i}']
    df_90['angle'] = df['angle'] + 90
    return df_90

# auxiliary function to flip samples
def flip(df):
    df_flip = df.copy()
    df_flip['mfc1'] = df['mfc2']
    df_flip['mfc2'] = df['mfc1']
    df_flip['mfc3'] = df['mfc4']
    df_flip['mfc4'] = df['mfc3']
    for i in [1, 2, 3]:
        df_flip[f'x{i}'] = 1 - df[f'x{i}']
        df_flip[f'y{i}'] = df[f'y{i}']
    df_flip['flipped'] = True
    return df_flip

# load data
df = pd.read_csv('data.csv', index_col=0)

# augment data
df['angle'] = 0
df = df.append(rotate90(df))
df = df.append(rotate90(df[(df['angle'] == 90)]))
df = df.append(rotate90(df[(df['angle']==180)]))
df['flipped'] = False
df = df.append(flip(df))

# keep only 1-leakage samples (only these are considered in this paper)
df = df[df['n'] == 1]

# normalize total flow so that x_1 + x_2 + x_3 + x_4 = 1
df['mfcsum'] = df['mfc1'] + df['mfc2'] + df['mfc3'] + df['mfc4']
df[['mfc1', 'mfc2', 'mfc3', 'mfc4']] = df[['mfc1', 'mfc2', 'mfc3', 'mfc4']].div(df['mfcsum'], axis=0)

# remove obsolete columns
df.drop(['s1', 'x2', 'y2', 's2', 'x3', 'y3', 's3', 'n', 'mfcsum'], axis=1, inplace=True)

# apply coordinate transformation to get (y_1, y_2) in [-1, 1] x [-1, 1]
df['x1'] = df['x1'].map(lambda z: 2 * z - 1)
df['y1'] = df['y1'].map(lambda z: -2 * z + 1)

# get training and test data
X_train = df.loc[(df['split'] == 'train') & (df['angle'] == 0) & (df['flipped'] == False), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
X_train_augmented = df.loc[(df['split'] == 'train'), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
X_test = df.loc[(df['split'] == 'test') & (df['angle'] == 0) & (df['flipped'] == False), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
X_test_augmented = df.loc[(df['split'] == 'test'), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
Y_train = df.loc[(df['split'] == 'train') & (df['angle'] == 0) & (df['flipped'] == False), ['x1', 'y1']].values.astype(np.float32)
Y_train_augmented = df.loc[(df['split'] == 'train'), ['x1', 'y1']].values.astype(np.float32)
Y_test = df.loc[(df['split'] == 'test') & (df['angle'] == 0) & (df['flipped'] == False), ['x1', 'y1']].values.astype(np.float32)
Y_test_augmented = df.loc[(df['split'] == 'test'), ['x1', 'y1']].values.astype(np.float32)

# split validation data from training data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)
X_train_augmented, X_val_augmented, Y_train_augmented, Y_val_augmented = train_test_split(X_train_augmented, Y_train_augmented, test_size=0.1, random_state=0)

# custom hidden layer for equivariant model
class EquivariantHidden(keras.layers.Layer):
    def __init__(self):
        super(EquivariantHidden, self).__init__()
        
    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(stddev=0.2)
        self.a = self.add_weight(shape=(), initializer=initializer, trainable=True)
        self.b = self.add_weight(shape=(), initializer=initializer, trainable=True)
        self.c = self.add_weight(shape=(), initializer=initializer, trainable=True)
        
    def call(self, inputs):
        return tf.nn.elu(self.a * inputs + \
                         self.b * (tf.gather(inputs, [3, 2, 1, 0], axis=1) + tf.gather(inputs, [1, 0, 3, 2], axis=1)) + \
                         self.c * tf.gather(inputs, [2, 3, 0, 1], axis=1))
    
# custom output layer for equivariant model
class EquivariantOutput(keras.layers.Layer):
    def __init__(self):
        super(EquivariantOutput, self).__init__()
        
    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(stddev=0.2)
        self.d = self.add_weight(shape=(), initializer=initializer, trainable=True)
        
    def call(self, inputs):
        return self.d * tf.concat([tf.reduce_sum(inputs * [1, -1, -1, 1], axis=1, keepdims=True),
                                   tf.reduce_sum(inputs * [-1, -1, 1, 1], axis=1, keepdims=True)], axis=1)
    
# initialize dataframe for results
results = pd.DataFrame(columns=['type', 'augmented', 'depth', 'width', 'epochs', 'batch_size', 'learning_rate', 'mse_train', 'mse_train_aug', 'mse_val', 'mse_val_aug', 'mse_test', 'mse_test_aug'])

# set hyperparams and train
epochs = 1000
batch_size = 32

for learning_rate in [1e-4, 1e-3, 1e-2]:
    for depth in range(1, 11):
        for _ in range(10):

            # train equivariant model on original data
            model = keras.models.Sequential([EquivariantHidden() for _ in range(depth)])
            model.add(EquivariantOutput())
            model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
            early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=early_stopping, verbose=0)
            mse_train = model.evaluate(X_train, Y_train, verbose=0)
            mse_train_aug = model.evaluate(X_train_augmented, Y_train_augmented, verbose=0)
            mse_val = model.evaluate(X_val, Y_val, verbose=0)
            mse_val_aug = model.evaluate(X_val_augmented, Y_val_augmented, verbose=0)            
            mse_test = model.evaluate(X_test, Y_test, verbose=0)
            mse_test_aug = model.evaluate(X_test_augmented, Y_test_augmented, verbose=0)
            results_tmp = np.array(['eqv', False, depth, 4, epochs, batch_size, learning_rate, mse_train, mse_train_aug, mse_val, mse_val_aug, mse_test, mse_test_aug]).reshape(1, -1)
            results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)

            # train equivariant model on augmented data
            model = keras.models.Sequential([EquivariantHidden() for _ in range(depth)])
            model.add(EquivariantOutput())
            model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
            early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
            model.fit(X_train_augmented, Y_train_augmented, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=early_stopping, verbose=0)
            mse_train = model.evaluate(X_train, Y_train, verbose=0)
            mse_train_aug = model.evaluate(X_train_augmented, Y_train_augmented, verbose=0)
            mse_val = model.evaluate(X_val, Y_val, verbose=0)
            mse_val_aug = model.evaluate(X_val_augmented, Y_val_augmented, verbose=0)
            mse_test = model.evaluate(X_test, Y_test, verbose=0)
            mse_test_aug = model.evaluate(X_test_augmented, Y_test_augmented, verbose=0)
            results_tmp = np.array(['eqv', True, depth, 4, epochs, batch_size, learning_rate, mse_train, mse_train_aug, mse_val, mse_val_aug, mse_test, mse_test_aug]).reshape(1, -1)
            results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)

            for width in [4, 16]:
                # train fcnn on original data
                model = keras.models.Sequential([keras.layers.Dense(width, activation='elu') for _ in range(depth)])
                model.add(keras.layers.Dense(2))
                model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
                early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
                model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=early_stopping, verbose=0)
                mse_train = model.evaluate(X_train, Y_train, verbose=0)
                mse_train_aug = model.evaluate(X_train_augmented, Y_train_augmented, verbose=0)
                mse_val = model.evaluate(X_val, Y_val, verbose=0)
                mse_val_aug = model.evaluate(X_val_augmented, Y_val_augmented, verbose=0) 
                mse_test = model.evaluate(X_test, Y_test, verbose=0)
                mse_test_aug = model.evaluate(X_test_augmented, Y_test_augmented, verbose=0)
                results_tmp = np.array(['fcnn', False, depth, width, epochs, batch_size, learning_rate, mse_train, mse_train_aug, mse_val, mse_val_aug, mse_test, mse_test_aug]).reshape(1, -1)
                results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)

                # train fcnn on augmented data
                model = keras.models.Sequential([keras.layers.Dense(width, activation='elu') for _ in range(depth)])
                model.add(keras.layers.Dense(2))
                model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
                early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
                model.fit(X_train_augmented, Y_train_augmented, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=early_stopping, verbose=0)
                mse_train = model.evaluate(X_train, Y_train, verbose=0)
                mse_train_aug = model.evaluate(X_train_augmented, Y_train_augmented, verbose=0)
                mse_val = model.evaluate(X_val, Y_val, verbose=0)
                mse_val_aug = model.evaluate(X_val_augmented, Y_val_augmented, verbose=0) 
                mse_test = model.evaluate(X_test, Y_test, verbose=0)
                mse_test_aug = model.evaluate(X_test_augmented, Y_test_augmented, verbose=0)
                results_tmp = np.array(['fcnn', True, depth, width, epochs, batch_size, learning_rate, mse_train, mse_train_aug, mse_val, mse_val_aug, mse_test, mse_test_aug]).reshape(1, -1)
                results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)
        
            results.to_csv('results.csv', index=False)
