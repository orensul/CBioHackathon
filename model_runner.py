import pomegranate
import numpy as  np
from pomegranate import (
    DiscreteDistribution,
    HiddenMarkovModel,
    State,
)
from possible_observations import possible_observations
from fetch_pdbtm_db import (
    read_training_file
)
from pomegranate_model import create_model



import itertools

def get_num_tm(s):
    positions = []
    for i in range(len(s)):
        if s[i] == 'I':
            positions.append(i)
    return len(list(to_ranges(positions)))


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]






def main():
    model = create_model()
    observation, labels = read_training_file('training_data.txt')
    observation, labels = np.array(observation), np.array(labels)
    print(len(observation), len(labels), type(observation))

    indices = np.arange(0, len(observation))
    # np.random.shuffle(indices)
    train_indices, test_indices = indices[:500], indices[50:]

    m, history = model.fit(
        observation[train_indices],
        labels=labels[train_indices],
        return_history=True,
        max_iterations=10,
        lr_decay=0.001,
        batches_per_epoch=32,
    )
    import json
    with open('model_json_100sam_labeled', 'w') as f:
        json.dump(model.to_json(), f)

    transition_matrix = model.dense_transition_matrix
    transition_matrix()
    states = np.array([state.name for state in model.states])

    # print(model.log_probability(observation[0]))
    # print(history)
    # print(states)
    # print(states[model.predict(observation[0])])

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
    print("avg acc: " + str(round(np.mean(acc_np_array), 2)))
    print("var acc: " + str(round(np.var(acc_np_array), 2)))

    count_correct, count_total = 0, 0
    for index in range(len(test_indices)):
        prediction_list = states[model.predict(observation[test_indices[index]])]
        binary_prediction_list = ['O' if item == 'B' else 'I' for item in prediction_list[1:-1]]
        prediction_str = ''.join(binary_prediction_list)
        ground_truth_list = labels[test_indices[index]]
        binary_ground_truth_list = ['O' if item == 'B' else 'I' for item in ground_truth_list[1:-1]]
        ground_truth_str = ''.join(binary_ground_truth_list)

        num_tm_ground_truth = get_num_tm(ground_truth_str)
        num_tm_prediction = get_num_tm(prediction_str)

        if num_tm_ground_truth == num_tm_prediction:
            count_correct += 1
        count_total += 1

    print(round(count_correct / count_total * 100, 2))
























if __name__ == '__main__':
    main()