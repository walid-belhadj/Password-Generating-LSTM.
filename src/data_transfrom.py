# -*- coding: utf-8 -*-

import numpy as np

"""
generer pour chaque longueur de mot de passe un dataset 
"""
# Liste de tous les caractères ( 95 caractères)
all_characters = [
	'1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
	'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd',
	'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm',
	'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
	'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M',
	',', '.', '/', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(',
	')', '_', '+', '<', '>', '?', ' ', '{', '}', '|', ':', '\"', '[',
	']', '\\', '\'', ';', '`', '-', '='
]
#	transformer les caractères sous la forme de vecteurs
def char_to_vector(c, tables=all_characters, start=False, end=False):
	"""
		convertir character to 1-of-N codage
	"""
	#attribuer au caractère 'W' la val '1' si on le croise, le reste la val '0'
	vec_len = len(tables) + 2
	vector =[0 for _ in range(vec_len)]
	# character de fin
	if start:
		vector[0] = 1
		return vector
	# caractère de fin
	if end:
		vector[-1] = 1
		return vector
	vector[tables.index(c)+1] = 1
	return vector # return le vecteur

# convertir le mot passe à une réprésentation par vecteur
def password_to_vector(password, width=3):
	# tester la taille de mot de passe à titre indicatif
	if len(password) + 2 < width + 1:
		raise Exception('Mot de passe trop petit')

	X = []
	Y = []
# construction de la X et de Y de sorte que x represente le charecteur et y le charactère à prédire
	x0 =[ char_to_vector(None, start=True) ]
	for c in password[:width-1]:
		x0.append(char_to_vector(c))
	y0 = char_to_vector(password[width-1])
	X.append(x0)
	Y.append(y0)
# contruction des mots de passe
	for i in range(len(password)-width):
		x = [ char_to_vector(c) for c in password[i:i+width] ]
		y = char_to_vector(password[i+width])
		X.append(x)
		Y.append(y)

	x1 = [] # table temporaire
	for c in password[-width:]:
		x1.append(char_to_vector(c))
	y1 = char_to_vector(None, end=True)
	X.append(x1)
	Y.append(y1)
	return X, Y

# transformation du coprus vers des dataset sous l'extension .npz de la lib 'numpy'
def transform_dataset(seq_len):
	with open('../data/file.txt', 'r') as f:
		X = []
		Y = []
		# parcourir les tuples ( vecteur, password )
		for i, pw in enumerate(f):
			try:
				# rstrip supprimer l'espacement à la fin
				x, y = password_to_vector(pw.rstrip(), width=seq_len)
				# retourne le mot de passe (x) avec sa longueur ( tout dépend de seq_lenth ( y)
			except Exception:
				print('[ligne %d] longueur:  < %d' % (i+1, seq_len))
			else: # incrémente au prochain tuplet ( mdp, vecteur)
				X += x
				Y += y
		X = np.array(X) # sauvegarder mot de passe sous une liste en X
		Y = np.array(Y) # sauvegarder en Y le vecteur correspondant
		#print(X.shape, Y.shape)
		np.savez('../data/dataset_%d'%seq_len, X=X, Y=Y)
def main():
	for element in range(1,16):
		transform_dataset(seq_len=element)

if __name__ == '__main__':
	main()




"""
def vector_to_char(vec, tables=all_characters):
	if vec[-1] == 1:
		return '@'
	if vec[0] == 1:
		return '&'
	for i, e in enumerate(vec):
		if e == 1:
			return tables[i-1]
"""