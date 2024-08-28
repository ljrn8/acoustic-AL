from tensorflow import keras
from tensorflow.keras import layers


def CRNN(input_shape, num_classes, n_filters):
    freq_len, time_len, _ = input_shape

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN layers
    for filters in n_filters:
        model.add(
            layers.Conv2D(
                filters, (3, 3), activation="relu", padding="same", strides=(2, 1)
            )
        )
        model.add(layers.MaxPooling2D((2, 1)))  # frequency pooling only

    model.add(layers.Flatten())
    model.add(
        layers.Reshape((time_len, -1))
    )  # input matrix for RNN shape=(time_frames, new CNN features)

    # RNN layers
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


def CNN(input_shape, n_filters):
    freq_len, time_len, _ = input_shape

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i, filters in enumerate(n_filters):
        model.add(
            layers.Conv2D(
                filters, (3, 3), activation="relu", padding="same", strides=(2, 1)
            )
        )
        model.add(layers.MaxPooling2D((2, 1)))  # frequency pooling only

    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(time_len))
    return model  
    
    
def CRNN_flat(input_shape, n_filters):
    freq_len, time_len, _ = input_shape

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN layers
    for filters in n_filters:
        model.add(
            layers.Conv2D(
                filters, (3, 3), activation="relu", padding="same", strides=(2, 1)
            )
        )
        model.add(layers.MaxPooling2D((2, 1)))  # frequency pooling only

    # model.add(layers.Flatten())
    model.add(
        layers.Reshape((time_len, -1))
    )  # input matrix for RNN shape=(time_frames, new CNN features)

    # RNN layers
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dense(1, activation="sigmoid"))
    # model.add(layers.Flatten())

    return model

if __name__ == "__main__":
    model = CRNN_flat((512, 127, 1), 
                      n_filters=[32, 64, 128, 256])
    print(model.summary())
    
    
