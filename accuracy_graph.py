import pomegranate
import numpy as np
from fetch_pdbtm_db import (
    read_training_file
)
import json
from collections import defaultdict
import matplotlib.pyplot as plt

json_file = open('saved_model', 'r')
loaded_model_json = json.load(json_file)
json_file.close()
model = pomegranate.hmm.HiddenMarkovModel.from_json(loaded_model_json)


num_training_samples = 2000
test_training_ratio = 0.3
num_test_samples = int(num_training_samples * test_training_ratio)
training_file_name = 'training_data.txt'



observation, labels = read_training_file(training_file_name)
observation, labels = np.array(observation), np.array(labels)
print("number of observations: " + str(len(observation)))
print("number of labels: " + str(len(labels)))
print("number of training samples: " + str(num_training_samples))
print("number of test samples: " + str(num_test_samples))

states = np.array([state.name for state in model.states])
indices = np.arange(0, len(observation))
test_indices = indices[:num_test_samples]

length_to_score = defaultdict(list)

num_of_trans_membrane_to_acc = defaultdict(list)

for index in range(len(test_indices)):
    prediction_list = states[model.predict(observation[test_indices[index]])]
    binary_prediction_list = ['O' if item == 'B' else 'I' for item in prediction_list[1:-1]]
    prediction_str = ''.join(binary_prediction_list)
    ground_truth_list = labels[test_indices[index]]
    binary_ground_truth_list = ['O' if item == 'B' else 'I' for item in ground_truth_list[1:-1]]
    ground_truth_str = ''.join(binary_ground_truth_list)
    pred_membrane = prediction_str.split('O')
    truth_membrane = ground_truth_str.split('O')
    num_in_pred = len(list(filter(lambda x: x != '', pred_membrane)))
    num_in_truth = len(list(filter(lambda x: x != '', truth_membrane)))
    print(num_in_pred/num_in_truth)
    length_to_score[len(ground_truth_str)].append(num_in_pred/num_in_truth)
    num_of_trans_membrane_to_acc[num_in_truth].append(num_in_pred/num_in_truth)

lengths = []
accs = []

trans = []
accs2 = []

for l in sorted(length_to_score.keys()):
    lengths.append(l)
    accs.append(sum(length_to_score[l])/len(length_to_score[l]))

# for n in sorted(num_of_trans_membrane_to_acc.keys()):
#     trans.append(n)
#     accs2.append(sum(num_of_trans_membrane_to_acc[n])/len(num_of_trans_membrane_to_acc[n]))

lengths= np.array(lengths)
plt.plot(lengths, accs)

# m, b = np.polyfit(lengths, accs, 1)

# plt.plot(lengths, m*lengths + b)
# plt.title('Number of Motifs Accuracy Rate by Length of the Sequence')
plt.xlabel('Length of the Sequence')
plt.ylabel('Number of Motifs Accuracy Rate')
plt.show()

# trans= np.array(trans)
# plt.plot(trans, accs2)
#
#
# # plt.title('Number of motifs accuracy rate by number of motifs')
# plt.xlabel('Number of Motifs')
# plt.ylabel('Number of Motifs Accuracy Rate')
# plt.show()