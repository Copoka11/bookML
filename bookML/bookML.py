from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
data = load_iris()

features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

print(target)
print(target_names)
#print(features, features.size)


for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'o'
    elif t == 2:
        c = 'b'
        marker = 'x'
    plt.scatter(features[target == t, 2],
                features[target == t, 3],
                marker=marker,
                c=c)

labels = target_names[target]
plength = features[:, 2]                    #petal lenth is feature pos. 2
is_setosa = (labels == "setosa")            #create boolean massive
max_setosa = plength[is_setosa].max()       #
min_non_setosa = plength[~is_setosa].min()

print('maximum of setosa: {0}.'.format(max_setosa))
print('minimum of others: {0}'.format(min_non_setosa))

plt.show()

plt.cla()
plt.clf()


####################################################################################

features = features[~is_setosa]
target = target[~is_setosa]
#print(features, features.size)
labels = labels[~is_setosa]
#print(labels, labels.size)

is_virginica = (labels == 'virginica')          #bool massive

best_acc = -1.0
#print(features.shape[1])                       #4


print(features)
print(target)



for fi in range(features.shape[1]):                 #0 1 2 3 
    thresh = features[:, fi]                        #rows 0 1 2 3 
#   print(thresh)
    for t in thresh:                                #for row
        feature_i = features[:, fi]                 #all in row
        pred = (feature_i > t)                      #bool, True if current item (t) less than item in the actual row 
        acc = (pred == is_virginica).mean()         #совпадение порога с вирджиникой
        rev_acc = (pred == ~is_virginica).mean()    #с не вирджиникой
        if rev_acc > acc:                           #условие хранит понимание порога вирджиники или не вирджиники?
            reverse = True
            acc = rev_acc
        else:
            reverse = False

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse

print('accuracy: ', best_acc, ', feature number: ', best_fi, ', feature currency: ',best_t)

def is_virginica_test (fi, t, reverse, example):
    "Apply threshold model to a new example"
    test = example[fi] > t
    if reverse:
        test = not test
    return test


for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'x'
    elif t == 2:
        c = 'b'
        marker = 'o'
    plt.scatter(features[target == t, 3],
                features[target == t, 2],
                marker=marker,
                c=c)

plt.vlines(best_t, 0, 7)
plt.show()
"""
correct = 0.0
for ei in range(len(features)):
    training = np.ones(len(features), bool)
    training[ei] = false
    testing = ~training
    model =
    

"""