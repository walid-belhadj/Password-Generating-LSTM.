import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tokenizer import tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 25
BATCH_SIZE = 64
VAL_SPLIT = 0.3 # 0.7 test
SEQ_RANGE = [1,15]
max_len=15
def model_to_predicted(max_len=15):
    for seq_len in range(SEQ_RANGE[0], SEQ_RANGE[1] + 1):
        dataset = np.load('../data/dataset_%d.npz' % seq_len)
        X, Y = dataset['X'], dataset['Y']
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        loaded_model = tf.keras.models.load_model('../model/%s_model_%d.h5' %seq_len)
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
    for ele in range(1,15):
        # model_train(model_type)
        model_to_predicted(ele)
if __name__ == '__main__':
    main()

