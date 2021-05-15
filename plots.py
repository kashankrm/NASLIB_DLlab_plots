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
    plt.scatter([pred]*3,best_val_acc[i,:])
plt.title("Best Validation Accuracy for 3 trial")
plt.tight_layout()

plt.subplot(3,1,2)
test_acc = [d["test_acc"] for d in data.values()]
best_test_acc = np.array(test_acc).max(axis=2)
for i,pred in enumerate(predictors):
    plt.scatter([pred]*3,best_test_acc[i,:])
plt.title("Best Test Accuracy for 3 trial")
plt.tight_layout()

plt.subplot(3,1,3)
train_acc = [d["train_acc"] for d in data.values()]
best_train_acc = np.array(train_acc).max(axis=2)
for i,pred in enumerate(predictors):
    plt.scatter([pred]*3,best_train_acc[i,:])
plt.title("Best Train Accuracy for 3 trial")
plt.tight_layout()


plt.show()


