import tensorflow as tf
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def create_model(input_shape: tuple, lr: float):
    """
    Creates our DNN Model.

    Parameters
    ----------
    input_shape : tuple

    lr : float
        Learning rate

    Returns
    -------
    MLP model

    """
    regressor = Sequential()
    regressor.add(Flatten(input_shape=input_shape))
    regressor.add(Dense(units=160, activation='relu'))
    regressor.add(Dense(units=180, activation='relu'))
    regressor.add(Dense(units=160, activation='relu'))
    regressor.add(Dense(units=240, activation='relu'))
    regressor.add(Dense(units=1))
    regressor.summary()
    Adam(learning_rate=lr)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor


def get_callbacks(central):
    cbs = list()
    checkpoint_path = f"data/models/{central}/model_{central}.ckpt"
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=False,
                                            period=2)
    cbs.append(cp_callback)
    # es_callback = callbacks.EarlyStopping(monitor='loss')
    # cbs.append(es_callback)
    return cbs


def load_model(regressor, central):
    latest = tf.train.latest_checkpoint(f"data/models/{central}")
    regressor.load_weights(latest)
