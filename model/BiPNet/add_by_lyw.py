import pickle
dataset_name="Grocery_and_Gourmet_Food"
test_data :tuple = pickle.load(open('./datasets/' + dataset_name + '/test.txt', 'rb'))

for i in range(len(test_data)):
    for j in range(len(test_data[0])):
        for k in range(len(test_data[0][0])):
            print(i,j,k)