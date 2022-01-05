file1 = open("mdp-générés2.txt", "rt")
file2 = open("../data/eval.txt", "rt")
data1 = file1.read()
data2 = file2.read()
words1= data1.split()
words2 = data2.split()

print('Number of words in text file :', len(words1))
print('Number of words in text file :', len(words2))

with open("mdp-générés2.txt.txt", 'r') as file1:
    with open("../data/eval.txt", 'rt') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

#renvoyé les mot de passe identique
with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)


