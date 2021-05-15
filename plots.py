import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_dir = "RE_predictor_dir_0/cifar10/nas_predictors/nasbench201/"
if os.name == 'nt':
    base_dir = base_dir.replace('/','\\')
def get_data_from_pred_dir(predictor):
    test_acc = []
    train_acc = []
    val_acc = []
    data_dir = os.path.join(base_dir,predictor)
    for trial in ['0','1','2']:
        data_file = os.path.join(data_dir,trial,"errors.json")
        with open(data_file) as f:
            data = json.load(f)
        train_acc.append(data[1]["train_acc"])
        test_acc.append(data[1]["test_acc"])
        val_acc.append(data[1]["valid_acc"])
    return np.array(train_acc),np.array(val_acc),np.array(test_acc)

def get_median(arr):
    return [np.median(x) for x in arr]
def get_round_robin_chunks(iterable, num_chunks):
    l = []
    for i in range(num_chunks):
        l.append([])
    count = 0
    for i in iterable:
        l[count % num_chunks].append(i)
        count+=1
    return l

predictors = os.listdir(base_dir)

data = {}
for pred in predictors:
    train_acc, val_acc, test_acc = get_data_from_pred_dir(pred)
    data[pred] = {"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc}

# plot best test_acc
plt.subplot(3,1,1)
val_acc = [d["val_acc"] for d in data.values()]
best_val_acc = np.array(val_acc).max(axis=2)
for i,pred in enumerate(predictors):
    x=[pred]
    y = best_val_acc[i,:].mean()
    yerr = best_val_acc[i,:].max() - best_val_acc[i,:].min() 
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.')
plt.title("Best Validation Accuracy for 3 trial")
plt.tight_layout()

plt.subplot(3,1,2)
test_acc = [d["test_acc"] for d in data.values()]
best_test_acc = np.array(test_acc).max(axis=2)
for i,pred in enumerate(predictors):
    x=[pred]
    y = best_test_acc[i,:].mean()
    yerr = best_test_acc[i,:].max() - best_test_acc[i,:].min() 
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.')
plt.title("Best Test Accuracy for 3 trial")
plt.tight_layout()

plt.subplot(3,1,3)
train_acc = [d["train_acc"] for d in data.values()]
best_train_acc = np.array(train_acc).max(axis=2)
for i,pred in enumerate(predictors):
    x=[pred]
    y = best_train_acc[i,:].mean()
    yerr = best_train_acc[i,:].max() - best_train_acc[i,:].min() 
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.')
plt.title("Best Train Accuracy for 3 trial")
plt.tight_layout()
plt.show()



plt.title("Validation accuracy")
plt.xlabel("epcohs")
num_predictor= len(predictors)
idxs = get_round_robin_chunks(range(300),num_predictor)
for i, pred in enumerate(predictors):
    idx = idxs[i]
    val_acc = data[pred]['val_acc']
    val_acc_mean = val_acc.mean(axis=0)
    val_acc_err = val_acc.max(axis=0) - val_acc.min(axis=0) 
    x = np.arange(0,300)[idx]
    y = val_acc_mean[idx]
    yerr = val_acc_err[idx]
    
    plt.ylabel("%")
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.',elinewidth=0.5,label=pred)
plt.legend()    
plt.show()

plt.title("Test accuracy")
plt.xlabel("epcohs")
num_predictor= len(predictors)
idxs = get_round_robin_chunks(range(300),num_predictor)
for i, pred in enumerate(predictors):
    idx = idxs[i]
    test_acc = data[pred]['test_acc']
    test_acc_mean = test_acc.mean(axis=0)
    test_acc_err = test_acc.max(axis=0) - test_acc.min(axis=0) 
    x = np.arange(0,300)[idx]
    y = test_acc_mean[idx]
    yerr = test_acc_err[idx]
    
    plt.ylabel("%")
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.',elinewidth=0.5,label=pred)
plt.legend()    
plt.show()

plt.title("Train accuracy")

plt.xlabel("epcohs")
num_predictor= len(predictors)
idxs = get_round_robin_chunks(range(300),num_predictor)
for i, pred in enumerate(predictors):
    idx = idxs[i]
    train_acc = data[pred]['train_acc']
    train_acc_mean = train_acc.mean(axis=0)
    train_acc_err = train_acc.max(axis=0) - train_acc.min(axis=0) 
    x = np.arange(0,300)[idx]
    y = train_acc_mean[idx]
    yerr = train_acc_err[idx]
    
    plt.ylabel("%")
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.',elinewidth=0.5,label=pred)
plt.legend()    
plt.show()


