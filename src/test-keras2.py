# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from datetime import datetime

# loader le corpus train
filename = "../data/train.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
# traitement : miniscules
raw_text = raw_text.lower()
# associer les carachtères à leurs équivalents en int
# creation mappage
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# resumé du contenu du corpus
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Nombre total des caractères: ", n_chars)
print ("Total Vocabulaire: ", n_vocab)

# préparer la dataset d'entré pour celle de sortie paire codé en entier
seq_length = 10 #taille de séuence ; combien je traite à la fois
dataX = []
dataY = []
# construction des vecteurs et des pathernes entré/ sortie
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print ("Nombre total des pathernes: ", n_patterns)

# reshape X pour qu'il soit [échantillons, pas de temps, fonctionnalités]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normaliser
X = X / float(n_vocab)
# econder à chaud la variable de sortie
y = np_utils.to_categorical(dataY)

# definir le model LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# phase de compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# definir le checkpoint, point de repère pour un best modèle
filepath = "../model/best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir="..\\logs\\fit\\"+datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True)
callbacks_list = [checkpoint, tensorboard]

# fit le model
model.fit(X, y, epochs=25, batch_size=64, callbacks=callbacks_list, validation_split=0.3)