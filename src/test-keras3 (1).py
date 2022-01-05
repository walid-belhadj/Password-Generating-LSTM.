import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

# loader le corpus train
filename = "../data/train.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
# traitement : miniscules
raw_text = raw_text.lower()
# associer les carachtères à leurs équivalents en int
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# resumé du contenu du corpus
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Nombre total des caractères: ", n_chars)
print ("Total Vocabulaire: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 10
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X à samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalisation
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)
# genrer le model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# upoloader le best model
filename = "../model/best.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accurancy')
# on prends une seed aléatoirement
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate des mots de passe

for i in range(1000):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x / float(n_vocab)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = int_to_char[index]
  seq_in = [int_to_char[value] for value in pattern]
  #sys.stdout.write(result)

  pattern.append(index)
  pattern = pattern[1:len(pattern)]
  file3 = open("predicted_password_1000.txt", 'a+')
  file3.write(str(result))
    #print("Génération de 9000 mots de passe, sauvegardés dans:", "predicted_password_%d.txt" %pred_size )

print("generating 1000 passwords: ")
#gen_pred_pwd(1000)
#print("generating 100000  passwords: ")
#gen_pred_pwd(1000)
#print("generating 1000000  passwords: ")
#gen_pred_pwd(1000)
