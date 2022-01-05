


def modelCreation(self):
    training_x = np.array(self.doc_x)
    training_y = np.array(self.doc_y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(training_y[0]), activation='softmax'))

    model.summary()