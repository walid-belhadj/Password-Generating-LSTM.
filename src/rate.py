


file1 = open("../generatedpasswords/rockyou_small.txt", "rt")
file2 = open("../generatedpasswords/rockyou_small1.txt", "rt")
data1 = file1.read()
data2 = file2.read()
words1= data1.split()
words2 = data2.split()

print('Number of words in text file :', len(words1))
print('Number of words in text file :', len(words2))

with open("../generatedpasswords/rockyou_small.txt", 'r') as file1:
    with open("../generatedpasswords/rockyou_small1.txt", 'rt') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)


