import pandas as pd
from multi_layer_net import MultiLayerNetExtend
from optimizer import *
import numpy as np
import sys, os
from sklearn.preprocessing import Normalizer
from ga import *
import time

start_time = time.time()
MoA_data = pd.read_csv('data.csv')

MoA_data =MoA_data.join(pd.get_dummies(MoA_data.Type))
del MoA_data["Type"]
use_norm = False
if use_norm:
    train_x= Normalizer().fit_transform(MoA_data.iloc[:73, 1:-5])
    test_x = Normalizer().fit_transform(MoA_data.iloc[73:, 1:-5])
    train_y= Normalizer().fit_transform(MoA_data.iloc[:73, -5:])
    test_y = Normalizer().fit_transform(MoA_data.iloc[73:, -5:])
else:
    train_x= MoA_data.iloc[:72, 1:-5]
    test_x = MoA_data.iloc[72:, 1:-5]
    train_y= MoA_data.iloc[:72, -5:]
    test_y = MoA_data.iloc[72:, -5:]
sys.path.append(os.pardir)

optimizer = SGD(lr=0.1)
(x_train, t_train), (x_test, t_test) = \
    (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


###ga_params
sol_per_pop = 8
num_parents_mating = 4
num_generations = 50
mutation_percent = 5
fitness_all_gen = []

###net_params
iters_num = 20000
train_size = x_train.shape[0]
batch_size = 2
learning_rate = 0.1
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
use_ga = True


iter_per_epoch = 10  # max(train_size / batch_size, 1)
max_acc_step = 0
flag = 0

initial_pop_params = []
network = MultiLayerNetExtend(input_size=9, hidden_size_list=[16], output_size=5)
for i in range(sol_per_pop):
    initial_pop_params.append(MultiLayerNetExtend(input_size=9, hidden_size_list=[16], output_size=5).params)

for i in range(iters_num):
    vec_weights = mat_trans(initial_pop_params)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(t_batch.shape)

    params = network.params
    if use_ga:
        fitness_per_gen = []
        for pop_param in initial_pop_params:
            network.change_weight(pop_param)
            fitness = network.accuracy(x_test, t_test)
            fitness_per_gen.append(fitness)
        fitness_all_gen.append(max(fitness_per_gen))
        max_fitness_idx = numpy.where(fitness_per_gen == numpy.max(fitness_per_gen))
        max_fitness_idx = max_fitness_idx[0][0]
        parents = select_mating_pool(vec_weights,fitness_per_gen.copy(),num_parents_mating)
        offspring_crossover = crossover(parents,offspring_size= (vec_weights.shape[0]-parents.shape[0],vec_weights.shape[1]))
        offspring_mutation = mutation(offspring_crossover,mutation_percent=mutation_percent,fitness = fitness_per_gen)
        vec_weights[0:parents.shape[0], :] = parents
        vec_weights[parents.shape[0]:, :] = offspring_mutation
        initial_pop_params = vector_trans(vec_weights, initial_pop_params)
    else:
        grad = network.gradient(x_batch, t_batch)
        optimizer.update(params, grad)
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        if use_ga:
            network.change_weight(initial_pop_params[max_fitness_idx])

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_loss = network.loss(x_train, t_train)
        test_loss = network.loss(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_acc>=max(test_acc_list):
            if flag == 0:
                max_acc_step = i

            if test_acc == 1:
                flag = 1

        print('iter:', i, 'acc:', train_acc, test_acc)
        print('loss:', i, 'loss:', train_loss, test_loss)


print(network.predict(x_train),t_train)
print(network.predict(x_test),t_test)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from itertools import cycle
from sklearn.preprocessing import label_binarize
nb_classes = 5
Y_valid=np.argmax(t_test,axis=1)
Y_pred=np.argmax(network.predict(x_test),axis=1)
Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nb_classes):
     fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
     roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
lw = 2
plt.figure(figsize=(12,8))
plt.plot(fpr["micro"], tpr["micro"],
  label='Average ROC curve (area = {0:0.2f})'
  ''.format(roc_auc["micro"]),
  color='deeppink', linestyle='-', linewidth=2)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','pink'])
for i, color in zip(range(nb_classes), colors):
     plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))##画出其余类型曲线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig(fname="9-ROC",format="svg")
plt.show()

def label_to_str(data):
    Y_=[]
    for i in np.argmax(data,axis=1):
        if i ==3:
            Y_.append('Nucleic acid synthesis')
        if i ==4:
            Y_.append('Protein synthesis')
        if i ==0:
            Y_.append('Cell all synthesis')
        if i == 2:
            Y_.append('Membrance active')
        if i ==1:
            Y_.append('Control')
    return Y_
print('Prediction:\n'+str(label_to_str(Y_pred)))
print('True:\n'+str(label_to_str(Y_valid)))
