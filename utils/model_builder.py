from tensorflow import keras
from keras import layers

def model_builder(hp):
    model = keras.Sequential()
    n_hidden_layers = hp.Int('n_hidden_layers', min_value=1, max_value=3, step=1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.add(layers.Flatten(input_shape=(32,32,3)))

    for i in range(n_hidden_layers):
        n_units = hp.Int(f'layer_{i}_units', min_value=24, max_value=480, step=24)
        #drop_rate = hp.Float(f'layer_{i}_drop', min_value=0.1, max_value=0.35, step=0.05)
        #regu_cons = hp.Choice(f'layer_{i}_regu_cons', values=[1e-2, 1e-3, 1e-4, 1e-5])


        model.add(layers.Dense(
            units=n_units,
            activation="relu",
            #kernel_regularizer=keras.regularizers.l2(regu_cons)
        ))
        #model.add(layers.Dropout(rate=drop_rate))

    model.add(layers.Dense(units=10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    return model
