import time
import mnist_loader 
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import csv
import numpy as np
import network 

###########read test dataset###############
def read_csv(filename):
    with open(filename, "r") as f_input:
        return [np.asarray(tuple(map(float, row))) for row in csv.reader(f_input)]

data = read_csv('mnist_test_20_label_last_row.csv')   #### can change the location of the input test dataset .csv file here, 20 pictures are converted to .csv file
###########read test dataset###############


###########convert test dataset into matrix that can be read by feedforward funtion ###############

vec = []
newRow = []
subArray = []
for row in data:
    print(len(row))
    for i in xrange(0,len(row)-1):
        vec.append(row[i])
        #print(vec)
        newRow.append(vec)
        vec = []
    #print(newRow)
    subArray.append(newRow)
    newRow = []
print(subArray[0][0][0])


newRow_l = []
subArray_l = []
for row in data:
    newRow_l.append(row[784])
    #print(newRow)
    subArray_l.append(newRow_l)
    newRow_l = []
    
print(subArray_l)

c = zip(subArray, subArray_l)   #### matrix c is the final that read by feedward funtion (through SGD funtion) in line 54
###########convert test dataset into matrix that can be read by feedforward funtion ###############

print(test_data[0][0][0])
print(np.shape(c[0][0]))

net = network.Network([784, 30 , 10])
net.SGD(training_data, 30, 10, 0.2, test_data=c)  #### can adjust learning rate to get the optimum output

start = time.time()
print(start)

end = time.time()
print(end)
print(end - start)
