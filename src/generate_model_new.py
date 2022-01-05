# -*-coding: utf-8 -*-
import os
import sys
from math import floor
import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras.layers import Dense, GRU, LSTM, Dropout, Bidirectional
from keras.models import Sequential
from keras.models import save_model

# set tensorflow log level
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 25
BATCH_SIZE = 128
VAL_SPLIT = 0.3
SEQ_RANGE = [2, 6]


def generate_model(rnn_type=LSTM, input_shape=None, output_shape=None):
    # RNN model
    model = Sequential()
    # Bidriectional RNN layer1
    model.add(Bidirectional(rnn_type(128, return_sequences=True),
                            input_shape=input_shape
                            )
              )
    # Bidriectional RNN layer2
    model.add(Bidirectional(rnn_type(128))
              )
    # Dropout layer
    model.add(Dropout(0.2))
    # Fully connected layer
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics= ['accuracy']

    )

    return model


def load_data(seq_len=1, test_per=0.3):
    dataset = np.load('../data/dataset_%d.npz' % seq_len)
    X, Y = dataset['X'], dataset['Y']
    idx = int(floor(len(X) * (1 - test_per)))
    X_train, X_test = X[:idx], X[idx:]
    Y_train, Y_test = Y[:idx], Y[idx:]
    return (X_train, Y_train), (X_test, Y_test)


def plot_history(result, model_type, seq_len):
    # accuarcy plots
    fig = plt.figure()
    plt.title('%s model accuracy with seq_len=%d' % (model_type, seq_len))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # plot traing accuracy
    acc = result.history['accuracy']
    plt.plot(range(1, len(acc) + 1), acc)
    # plot validate accuracy
    val_acc = result.history['val_accuracy']
    plt.plot(range(1, len(val_acc) + 1), val_acc)

    plt.legend(['train', 'evaluation'], loc='upper left')
    plt.savefig('../plots/acc-%s-model-%d.png' % (model_type, seq_len))
    plt.close('all')

    # loss plots
    fig = plt.figure()
    plt.title('%s model loss with seq_len=%d' % (model_type, seq_len))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plot traing loss
    loss = result.history['loss']
    plt.plot(range(1, len(loss) + 1), loss)
    # plot validate loss
    val_loss = result.history['val_loss']
    plt.plot(range(1, len(val_loss) + 1), val_loss)

    plt.legend(['train', 'evaluation'], loc='upper left')
    plt.savefig('../plots/loss-%s-model-%d.png' % (model_type, seq_len))
    plt.close('all')


#
def save_result(result, model_type, seq_len):
    with open('../results/result-%s-with_seq_len-%d.txt' % (model_type, seq_len), 'w+') as file:
        # accuracy
        # file.write('accuaray:\n')
        acc = result.history['accuracy']
        for item in acc:
            file.write(str(item) + ' ')
        file.write('\n')
        # validate accuracy
        # file.write('validate accuracy:\n')
        val_acc = result.history['val_accuracy']
        for item in val_acc:
            file.write(str(item) + ' ')
        file.write('\n')
        # loss
        # file.write('loss:\n')
        loss = result.history['loss']
        for item in loss:
            file.write(str(item) + ' ')
        file.write('\n')
        # validate loss
        # file.write('validate loss:\n')
        val_loss = result.history['val_loss']
        for item in val_loss:
            file.write(str(item) + ' ')
        file.write('\n')


def model_train(model_type='lstm'):
    if model_type == 'lstm':
        rnn_type = LSTM
    elif model_type == 'gru':
        rnn_type = GRU
    else:
        raise Exception('unkown RNN type')

    for seq_len in range(SEQ_RANGE[0], SEQ_RANGE[1] + 1):
        # load data
        (X_train, Y_train), (X_test, Y_test) = load_data(seq_len=seq_len)
        # create model
        model = generate_model(
            rnn_type=rnn_type,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=Y_train.shape[1]
        )
        # Training
        print('------Training (%s model with seq_len %d)-------' % (model_type, seq_len))
        result = model.fit(
            X_train, Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT
        )
        plot_history(result, model_type, seq_len)
        save_result(result, model_type, seq_len)

        print('------Testing (%s model with seq_len %d)-------' % (model_type, seq_len))
        loss, accuracy = model.evaluate(X_test, Y_test)
        print('test loss: ', loss)
        print('test accuracy: ', accuracy)
        # save model
        save_model(model, '../model/%s_model_%d.h5' % (model_type, seq_len))
        del model


def predected(rnn_type=LSTM, input_shape=None, output_shape=None):
    filename = "../data/dataset_%d.npz"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    seq_length = 100
    datasetX = []
    datasetY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        datasetX.append([char_to_int[char] for char in seq_in])
        datasetY.append(char_to_int[seq_out])
    n_patterns = len(datasetX)
    X = numpy.reshape(datasetX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(datasetY)

    model = Sequential()
    # Bidriectional RNN layer1
    model.add(Bidirectional(rnn_type(128, return_sequences=True),
                            input_shape=input_shape
                            )
              )
    # Bidriectional RNN layer2
    model.add(Bidirectional(rnn_type(128))
              )
    # Dropout layer
    model.add(Dropout(0.2))
    # Fully connected layer
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']

    )

    filename = "weights-improvement-50-0.8767-bigger.hdf5"
    model.load_weights(filename)
    start = numpy.random.randint(0, len(datasetX) - 1)

    pattern = datasetX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        file3 = open("mdp8.txt", 'a+')
        file3.write(str(result) + "\n " + "\n")

    print("\nDone.")


def main():
    model_types = ['lstm', 'gru']
    for model_type in model_types:
        model_train(model_type)



if __name__ == '__main__':
    main()