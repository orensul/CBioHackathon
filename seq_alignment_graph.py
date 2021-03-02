from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import pomegranate
import numpy as np
from fetch_pdbtm_db import (
    read_training_file
)
import json
from collections import defaultdict
import matplotlib.pyplot as plt
# al = pairwise2.align.globalms("BBBAAA", "AAACCC", 1, -1, -2, -2)
# print(format_alignment(*al[0]))

json_file = open('saved_model', 'r')
loaded_model_json = json.load(json_file)
json_file.close()
model = pomegranate.hmm.HiddenMarkovModel.from_json(loaded_model_json)

num_training_samples = 2000
test_training_ratio = 0.1
num_test_samples = int(num_training_samples * test_training_ratio)
training_file_name = 'training_data.txt'

length_to_score = defaultdict(list)


observation, labels = read_training_file(training_file_name)
observation, labels = np.array(observation), np.array(labels)
print("number of observations: " + str(len(observation)))
print("number of labels: " + str(len(labels)))
print("number of training samples: " + str(num_training_samples))
print("number of test samples: " + str(num_test_samples))

indices = np.arange(0, len(observation))
test_indices = indices[:num_test_samples]
states = np.array([state.name for state in model.states])
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

plt.plot(lengths, scores, 'ro')
plt.show()