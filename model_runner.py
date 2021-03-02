import numpy as np
from fetch_pdbtm_db import (
    read_training_file
)
from pomegranate_model import create_model
import json

num_training_samples = 2000
test_training_ratio = 0.1
num_test_samples = int(num_training_samples * test_training_ratio)
training_file_name = 'training_data.txt'


def main():
    model = create_model()
    observation, labels = read_training_file(training_file_name)
    observation, labels = np.array(observation), np.array(labels)
    print("number of observations: " + str(len(observation)))
    print("number of labels: " + str(len(labels)))
    print("number of training samples: " + str(num_training_samples))
    print("number of test samples: " + str(num_test_samples))

    indices = np.arange(0, len(observation))
    train_indices, test_indices = indices[num_test_samples:num_training_samples], indices[:num_test_samples]
    _, _ = model.fit(observation[train_indices], labels=labels[train_indices], return_history=True, max_iterations=50,)

    with open('saved_model', 'w') as f:
        json.dump(model.to_json(), f)

    transition_matrix = model.dense_transition_matrix
    transition_matrix()
    states = np.array([state.name for state in model.states])

    print("============= evaluation =============")
    print("============= hidden state vs. hidden state =============")

    acc_list = []
    pred_actual_list = []
    for index in range(len(test_indices)):
        prediction_list = states[model.predict(observation[test_indices[index]])]
        binary_prediction_list = ['O' if item == 'B' else 'I' for item in prediction_list[1:-1]]
        prediction_str = ''.join(binary_prediction_list)
        ground_truth_list = labels[test_indices[index]]
        binary_ground_truth_list = ['O' if item == 'B' else 'I' for item in ground_truth_list[1:-1]]
        ground_truth_str = ''.join(binary_ground_truth_list)
        pred_actual_list.append((prediction_str, ground_truth_str))
        count_correct, count_total = 0, 0
        for i in range(len(prediction_list[1:-1])):
            if binary_prediction_list[i] == binary_ground_truth_list[i]:
                count_correct += 1
            count_total += 1

        acc_list.append(round(count_correct / count_total * 100, 2))

    acc_np_array = np.array(acc_list)
    print("samples")
    for i in range(len(test_indices)):
        print("prediction sample index: " + str(i))
        print(pred_actual_list[i][0])
        print("actual sample index: " + str(i))
        print(pred_actual_list[i][1])
        print("accuracy per sample: " + str(acc_list[i]) + "%")
    print("avg acc: " + str(round(np.mean(acc_np_array), 2)))
    print("var acc: " + str(round(np.var(acc_np_array), 2)))
























if __name__ == '__main__':
    main()