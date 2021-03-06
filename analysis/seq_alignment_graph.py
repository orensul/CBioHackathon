from Bio import pairwise2
import pomegranate
import numpy as np
from data.fetch_pdbtm_db import (
    read_training_file
)
import json
from collections import defaultdict
import matplotlib.pyplot as plt
# al = pairwise2.align.globalms("BBBAAA", "AAACCC", 1, -1, -2, -2)
# print(format_alignment(*al[0]))

json_file = open('../model/saved_model', 'r')
loaded_model_json = json.load(json_file)
json_file.close()
model = pomegranate.hmm.HiddenMarkovModel.from_json(loaded_model_json)

num_training_samples = 2000
test_training_ratio = 0.2
num_test_samples = int(num_training_samples * test_training_ratio)
training_file_name = '../data/training_data.txt'

length_to_score = defaultdict(list)


observation, labels = read_training_file(training_file_name)
observation, labels = np.array(observation), np.array(labels)
# print("number of observations: " + str(len(observation)))
# print("number of labels: " + str(len(labels)))
# print("number of training samples: " + str(num_training_samples))
# print("number of test samples: " + str(num_test_samples))

indices = np.arange(0, len(observation))
test_indices = indices[-num_test_samples:]

train_indices= indices[num_test_samples:num_training_samples]
states = np.array([state.name for state in model.states])
# for index in range(400):
#   print("(\"",end='')
#   print(re.sub('\d', '',
#                ''.join(states[model.predict(observation[test_indices[index]])]).replace('start','').replace('end','').replace('S','').replace('L','')),end='')
#   print("\",\"",end='')
#   print(re.sub('\d', '',
#                ''.join(labels[test_indices[index]]).replace('start','').replace('end','').replace('S','').replace('L','')),end='')
#   print("\"),")


for index in range(len(test_indices)):
    prediction_list = states[model.predict(observation[test_indices[index]])]
    binary_prediction_list = ['O' if item == 'B' else 'I' for item in prediction_list[1:-1]]
    prediction_str = ''.join(binary_prediction_list)
    ground_truth_list = labels[test_indices[index]]
    binary_ground_truth_list = ['O' if item == 'B' else 'I' for item in ground_truth_list[1:-1]]
    ground_truth_str = ''.join(binary_ground_truth_list)
    score = pairwise2.align.globalms(prediction_str, ground_truth_str, 1, -1, -2, -2, score_only=True)
    length_to_score[len(ground_truth_str)].append(score)

scores = []
lengths = []

for l in sorted(length_to_score.keys()):
    scores.append(sum(length_to_score[l])/len(length_to_score[l]))
    lengths.append(l)
lengths= np.array(lengths)
plt.plot(lengths, scores, 'o')

m, b = np.polyfit(lengths, scores, 1)

plt.plot(lengths, m*lengths + b)
# plt.title('Sequence alignment score by length of the sequence')
plt.xlabel('Length of Sequence')
plt.ylabel('Sequence Alignment Score')
plt.show()