import random
import math
files = glob.glob(training_data + '/**/*.tsv', recursive=True)
glob.glob(training_data + '/**')
list_train_total = []
list_test_total = []
for folder in glob.glob(training_data + '/**'):
    list_tsv = glob.glob(folder + '/*.tsv')
    list_test = random.sample(list_tsv, math.floor(len(list_tsv)*30/100))
    print(len(list_tsv))
    print(len(list_test))
    print('------------')
    list_train = [x for x in list_tsv if x not in list_test]
    list_train_total.append(list_train)
    list_test_total.append(list_test)
    temp_train.append(len(list_tsv))
list_train_total = [item for sublist in list_train_total for item in sublist]
list_test_total = [item for sublist in list_test_total for item in sublist]