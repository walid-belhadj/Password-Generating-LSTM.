# encoding: utf-8
from collections import Counter
import re, os

import matplotlib.pyplot as plt
import numpy as np
import pylab
from pip._vendor.distlib.compat import raw_input


# - Retourne la longueur moyenne des passwords dans ce fichier


def calculmoy(file):
    liste = []
    global longueur_totale
    # La variable "lines" est une liste contenant toutes les lignes du fichier
    file = open(file, 'r')
    lines = file.readlines()
    # total = len(lines)
    # print(lines)
    for line in lines:
        # print(line.strip())
        long = len(line) - 1
        liste.append(long)
        # print(long)

    # liste.sort()
    somme = sum(liste)
    moy = somme / longueur_totale
    return round(moy, 3)
    # print("moyenne =", moy)
    # print("**************************************")
    # print(liste)


class obj:
    nb_car = 0
    nb_mot = 0


# Affiche nombre de mots avec son nombres de caracteres
def nbmotavecNcaract(file):
    file = open(file, 'r')
    lines = file.readlines()
    global longueur_totale

    x = obj()
    x.nb_car = len(lines[0])
    x.nb_mot = 1
    liste_m = []
    liste_m.append(x)

    t_mots = len(liste_m)

    for line in range(1, longueur_totale):
        exist = False
        for mot in range(t_mots):
            if (len(lines[line]) == liste_m[mot].nb_car):
                exist = True
                liste_m[mot].nb_mot += 1
        # print(exist)
        if (exist == False):
            y = obj()
            y.nb_car = len(lines[line])
            y.nb_mot += 1
            liste_m.append(y)
            t_mots = len(liste_m)

    print("> Le nombre de categories est : {} \t ".format(len(liste_m)))

    for mot in range(t_mots):
        print("> Le nombre de mots de {} caracteres est de : {} ".format(liste_m[mot].nb_car - 1, liste_m[mot].nb_mot))

    file.close()


# - Retourne le nombre de mots ayant uniquement des lettres
def total_words_with_letters(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()

    for line in lines:
        line2 = line.strip()
        y = line2.isalpha()
        if (y == True):
            count = count + 1
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 2)


# - Retourne les mots de passe uniquement avec des chiffres(digits)

def only_digits(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()
    for line in lines:
        line2 = line.strip()
        x = line2.isdigit()
        if (x == True):
            count = count + 1
    # print("nombre de mdp avec des chiffres = ")
    # print(count)
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 2)


# - Retourne le nombre de mots de passe avec uniquement des lettres et des chiffres uniq
def only_letters_and_digits(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()

    for line in lines:
        line2 = line.strip()
        z = line2.isalnum()
        x = line2.isdigit()
        y = line2.isalpha()
        if (z == True) and (x == False) and (y == False):
            # if(x == True) and (y == True) and (z== True):
            count = count + 1

    # print("nombre de mdp avec des lettres/chiffres = ")
    # print(count)
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 2)


# - Retourne les mots de passe avec des caracteres speciaux uniquement
def only_special_char(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()
    for line in lines:
        line2 = line.strip()
        x = line2.isalpha()
        y = line2.isdigit()
        z = line2.isalnum()
        if (x == False) and (y == False) and (z == False):
            count = count + 1

    # print(count)
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 4)


# - Retourne le nombre de mots ayant n ou plus caracteres
def n_char(n, file):
    count = 0
    global longueur_totale
    infile = open(file, 'r')
    for line in infile:
        wordslist = line.splitlines()
        if (len(wordslist[0]) >= n):
            count += 1
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 3)


# - Retourne la longueur minimum d'un mot parmi tous
def min_length(file):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # print "mot de passe le plus court en clair : " + min(content, key=len)
    return len(min(content, key=len))


# - Retourne la longueur maximum d'un mot parmi tous
def max_length(file):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # print "mot de passe le plus long en clair : " + max(content, key=len)
    return len(max(content, key=len))


# - Retourne les password contenant des lettres en minuscules
def total_words_with_lowerletters(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()

    for line in lines:
        line2 = line.strip()
        y = line2.islower()
        if (y == True):
            count = count + 1
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 2)


# - Retourne les password contenant des lettres en majuscules
def total_words_with_upperletters(file):
    count = 0
    global longueur_totale
    file = open(file, 'r')
    lines = file.readlines()

    for line in lines:
        line2 = line.strip()
        y = line2.isupper()
        if (y == True):
            count = count + 1
    moyenne = (float(count) / float(longueur_totale)) * 100
    return count, round(moyenne, 2)


# - Retourne le nombre d'element dans la liste content
# - Soit le nombre de mots de passe dans ce corpus
def number_of_words(file):
    count = 0
    global longueur_totale
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    longueur_totale = len(content)
    return len(content)


def same_password(n, file):
    global longueur_totale
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    same = Counter(content)
    # print same
    res = [element for element, occ in same.items() if occ == n]
    # print res
    moyenne = (float(len(res)) / float(longueur_totale)) * 100
    return len(res), round(moyenne, 3)


# -------------------------- MAIN --------------------------
#
print('')
print('> Analyse de Corpus !\n')
longueur_totale = 0
# nom_fichier = raw_input("> Merci de rensigner le nom du fichier servant de corpus : ")
nom_fichier = "../data/train.txt"
data = []
if os.path.isfile(nom_fichier):
    print('> Le nom du fichier qui servira de corpus est ' + nom_fichier + '\n')
    print(' ---------------------------------------------------------------------------------- ')

    print('> Nombre total de lignes (mdp) dans ce fichier         :', number_of_words(nom_fichier), '\n')

    print('> Nombre de passwords avec : lettres                      :', total_words_with_letters(nom_fichier)[0],
          '- soit :', total_words_with_letters(nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords avec : chiffres                     :', only_digits(nom_fichier)[0], '- soit :',
          only_digits(nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords avec : caracteres speciaux          :', only_special_char(nom_fichier)[0], '- soit :',
          only_special_char(nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords avec : lettres / chiffres           :', only_letters_and_digits(nom_fichier)[0],
          '- soit :', only_letters_and_digits(nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords avec : lettre en minuscules         :', total_words_with_lowerletters(nom_fichier)[0],
          '- soit :', total_words_with_lowerletters(nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords avec : lettre en majuscules         :', total_words_with_upperletters(nom_fichier)[0],
          '- soit :', total_words_with_upperletters(nom_fichier)[1], '% du corpus', '\n')

    print("#########################################################################")

    print('> Longueur minimum d\'un mot de passe parmi tous        :', min_length(nom_fichier), 'caracteres')
    print('> Longueur maximale d\'un mot de passe parmi tous       :', max_length(nom_fichier), 'caracteres')
    print('> Longueur moyenne des mots de passe dans ce fichier   :', calculmoy(nom_fichier), '\n')

    print("#########################################################################")
    nbmotavecNcaract(nom_fichier)

    print("#########################################################################")

    for i in range(23):
        print('> Nombre de passwords avec', i, ', caracteres ou plus :', n_char(i, nom_fichier)[0], '- soit :',
              n_char(i, nom_fichier)[1], '% du corpus')

    print('\n> Nombre de passwords apparaissant deux fois           :', same_password(2, nom_fichier)[0], '- soit :',
          same_password(2, nom_fichier)[1], '% du corpus')
    print('> Nombre de passwords apparaissant trois fois          :', same_password(3, nom_fichier)[0], '- soit :',
          same_password(3, nom_fichier)[1], '% du corpus')

    print('')

# Génération graphiques global
objects_global = ['Caracteres speciaux .', 'Lettres .', 'Chiffres .',
                  'Lettres ET Chiffres', 'Lettres en minuscules', 'Lettres en majuscules']

pourcentages_global = [only_special_char(nom_fichier)[1], total_words_with_letters(nom_fichier)[1],
                       only_digits(nom_fichier)[1], only_letters_and_digits(nom_fichier)[1],
                       total_words_with_lowerletters(nom_fichier)[1], total_words_with_upperletters(nom_fichier)[1]]
y_pos = np.arange(len(objects_global))
plt.barh(y_pos, pourcentages_global, align='center', alpha=0.5, edgecolor="black")
plt.yticks(y_pos, objects_global)
plt.xlabel('Pourcentages')
plt.title('Stastistiques sur le corpus ' + nom_fichier)
plt.tight_layout()
plt.savefig('global_stats.png')
plt.close()

data = [total_words_with_letters(nom_fichier)[0], only_digits(nom_fichier)[0], only_letters_and_digits(nom_fichier)[0],
        only_special_char(nom_fichier)[0]]
labels = ['lettres', 'chiffres', 'lettres/chiffres', 'caracteres spéciaux']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
plt.pie(data, labels=labels, colors=colors, shadow=True, startangle=90)
plt.axis("equal")
plt.title('Analyse > ' + nom_fichier)
plt.tight_layout()
plt.savefig('Graphe1.png')
# plt.show()
# On ferme le plot actuel pour que les valeurs ne se cumulent pas
plt.close()

objects_char = []
pourcentages_char = []
# Génération graphique nombre de caractères
for i in range(1, 16):
    objects_char.append('Plus de ' + str(i) + ' caracteres')
    pourcentages_char.append(n_char(i, nom_fichier)[1])

y_pos = np.arange(len(objects_char))
plt.barh(y_pos, pourcentages_char, align='center', alpha=0.5, edgecolor="black")
plt.yticks(y_pos, objects_char)
plt.xlabel('Pourcentages')
plt.title('Nombre de caracteres > ' + nom_fichier)
plt.tight_layout()
plt.savefig('number_of_char.png')
# plt.show()
plt.close()

# stats sur chaque caracteres
# nombre de min et maj