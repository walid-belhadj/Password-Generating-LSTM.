# -*-coding: utf-8 -*-
import os
import pickle
from math import floor
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, GRU, LSTM, Dropout, Bidirectional
from keras.models import Sequential
from keras.models import save_model

from keras_preprocessing.sequence import pad_sequences
from tokenizer import tokenizer
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 25
BATCH_SIZE = 64
VAL_SPLIT = 0.3
SEQ_RANGE = [1,15]


def generate_model(rnn_type=LSTM, input_shape=None, output_shape=None):
    model = Sequential()
    model.add(Bidirectional(rnn_type(256, return_sequences=True),input_shape=input_shape))
    model.add(Bidirectional(rnn_type(128)))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_data(seq_len=None, test_per=0.3):
    dataset = np.load('../data/dataset_%d.npz'%seq_len)
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
        model = generate_model(rnn_type=rnn_type,input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=Y_train.shape[1])
        # Training
        print('------Training (%s model with seq_len %d)-------' % (model_type, seq_len))
        result = model.fit(X_train, Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=VAL_SPLIT)
        plot_history(result, model_type, seq_len)
        save_result(result, model_type, seq_len)
        print('------Testing (%s model with seq_len %d)-------' % (model_type, seq_len))
        loss, accuracy = model.evaluate(X_test, Y_test)
        print('test loss: ', loss)
        print('test accuracy: ', accuracy)
        # save model
        save_model(model, '../model/%s_model_%d.h5' % (model_type, seq_len))
        del model
def model_to_predicted(max_len=15):
    for seq_len in range(SEQ_RANGE[0], SEQ_RANGE[1] + 1):
        dataset = np.load('../data/dataset_%d.npz' % seq_len)
        X, Y = dataset['X'], dataset['Y']
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        loaded_model = tf.keras.models.load_model('../model/%s_model_%d.h5' %  seq_len)
        with open('tokenizer.pickle', 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
        txt = dataset[EPOCHS]
        seq = loaded_tokenizer.texts_to_sequences([txt])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = loaded_model.predict_classes(padded)
        with open('../generatedpasswords/predicted.txt', 'a+') as file:
            file.write(pred)


def main():
    model_types = ['lstm']
    #model_types = ['lstm', 'gru']
    for model_type in model_types:
        model_train(model_type)
        #model_to_predicted(15)


if __name__ == '__main__':
    main()
