import torch
import torch.nn as nn
import string
import random
# import sys
# from torch.utils.tensorboard import SummaryWriter
import time
from pip._vendor.distlib.compat import raw_input

x = time.time()
y = time.time()


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


# parametres à varier
# output_size = 10
num_epochs = 1000  # nombre d'époques de test
batch_size = 1  # nombre de lots de varie de 1 à 500 ...etc nombre de lots de donnée
learning_rate = 0.001  # taux d'apprentissage, taile de pas à chaque itération plus il est petit plus il apprend bien pour gerer la précision
# plus yzid plus tnaqes la precsision

# input_size = 128 #Le nombre de fonctionnalités attendues dans l'entrée x
sequence_length = 250  # nombre de caracteres qu'il va prendre dans le fichier
hidden_size = 256  # Le nombre d'entités dans l'état caché h nobre de nouronne par chaque reseau
num_layers = 2  # Nombre de couches récurrentes. Par exemple, le réglage num_layers=2 signifierait nombre de couche entrée sorite
# l'empilement de deux LSTM pour former un LSTM empilé , le deuxième LSTM
# prenant les sorties du premier LSTM et calculant les résultats finaux. Par défaut : 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get characters from string.printable
all_characters = string.printable  # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# n_characters = len(all_characters) # nombre de caractères total

# récupérer le corpus (Note can be any text file: not limited to just names)
nom_fichier = raw_input("> Merci de rensigner le nomesm du fichier test servant de corpus : ")
file = open(nom_fichier).read()
"""
nom_fichier2 = raw_input("> Merci de rensigner le nom du fichier train servant de corpus : ")
file2 = open(nom_fichier2).read()
"""
# Lecture de tous les caracteres presents dans le fichier
# all_characters = ''
for car in file:
    if car not in all_characters:
        all_characters += car

n_characters = len(all_characters)

a = []
for i in all_characters:
    b = i.strip()  # enlever les espaces
    # print(b)
    a.append(b)
c = random.choice(a)
print(c)


# MNIST dataset
# Le codeur et le décodeur sont également définis de manière similaire,
# avec un paramètre supplémentaire de num_layers, qui indique le nombre de couches dans chaque LSTM.
class RNN(nn.Module):
    """
    The RNN model will be a RNN followed by a linear layer,
    i.e. a fully-connected layer
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size  # taille de la couche
        self.embed = nn.Embedding(input_size, hidden_size)
        # Une table de recherche simple qui stocke les intégrations d'un dictionnaire et d'une taille fixes.
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # assuming batch_first = True for RNN cells
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        # out, (hidden, cell) = self.gru(out.unsqueeze(1), hidden)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)
        # en dehors de la sortie, rnn nous donne également le hidden cell cela nous donne la
        # possibilité de le passer à
        # la cellule suivante si nécessaire ; nous n'en aurons pas besoin ici
        # car le nn.RNN a déjà calculé tous les pas de temps
        # pour nous. rnn_out sera de taille [batch_size, seq_len, hidden_size]

    def init_hidden(self, batch_size):
        """
          Initialize hidden cell states, assuming
          batch_first = True for RNN cells
          l'état caché initial dans RNN/LSTM, qui est h0 dans les formules.
          Pour chaque époque, nous devons réinitialiser un nouvel hidden state de débutant, c'est parce que pendant le test,
           notre modèle n'aura aucune information sur la phrase de test et aura un hidden state initial de zéro.
          """
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return hidden, cell


class Generator:
    # Characteristics initialiser les parametres
    def __init__(self):
        self.chunk_len = sequence_length  # nombre de caracteres qu'il va prendre dans le fichier
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.print_every = 50  # affichage des resultats toutes les 50 epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate

    def char_tensor(self,
                    string):  # Un torch.Tensor est une matrice multidimensionnelle contenant des éléments d'un seul type de données.
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    # generate mdp taile de prediction
    def generate(self, initial_str=c, predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str
        # optimizer adam
        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    def train(self):
        # appell de fonction rnn
        self.rnn = RNN(
            n_characters, self.hidden_size, self.num_layers, n_characters
        ).to(device)
        # creer un model  adam optmizer
        # L'algorithme d'optimisation Adam est une extension de la descente de gradient
        # Adam est utilisé à la place de la procédure classique de descente de gradient stochastique
        # #our mettre à jour les poids du réseau de manière itérative en fonction des données d'apprentissage.
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")
        for epoch in range(1, self.num_epochs + 1):

            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            file3 = open("mdp.txt", 'a+')

            if epoch % self.print_every == 0:
                print("###########")
                print(f"epoch: {epoch} ")

                print(f"Loss: {loss}")
                mdp = self.generate()
                print(mdp + "\n")

                file3.write(f"epoch: {epoch} " + "\n")
                file3.write(str(mdp) + "\n" + "\n")

                z = (time.time() - x)
                print(convert(z))


gennames = Generator()

gennames.train()

# Driver program
t = (time.time() - x)
print(convert(t))

