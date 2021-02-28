import numpy as np
from fetch_pdbtm_db import (
    get_seeds,
    read_chains,
    get_alpha_helix_subsequences,
    get_alpha_sequences
)
from hmm_tcm import baum_welch


def main():
    chains = read_chains('pdbtm')[:30]
    observation = get_alpha_sequences(chains)
    seeds = get_seeds(get_alpha_helix_subsequences(chains))
    # seeds parameters
    k = 10
    i = 0
    p = 0.1

    trans_prob = np.zeros((11, 11))
    trans_prob[0, 0] = 1 - p
    trans_prob[0, 1] = p
    for i in range(1, 10):
        trans_prob[i, i+1] = 1
    trans_prob[-1, 0] = 1

    em_prob = np.exp(seeds[k][0][1].to_numpy())

    # em_prob = np.empty((11, 22))
    # em_prob[:, :] = 1/22
    # states = ['B'] + [f'M{i}' for i in range(1, k + 1)]

    emission_matrix, transition_matrix, ll_hist, p = baum_welch(
        observation, em_prob, trans_prob, convergence_threshold=0.01
    )
    print(ll_hist)


if __name__ == '__main__':
    main()

