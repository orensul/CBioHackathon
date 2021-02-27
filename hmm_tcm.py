import numpy as np
from scipy.special import logsumexp
import math

from possible_observations import possible_observations
letters_dict = {v: i for i, v in enumerate(possible_observations)}


def create_transition_matrix(p, num_states):
    """
    Create the transition matrix(num_states, num_states) according to the ZOOPS model
    :param p: the probability of transition from a background state(b1, b2) to the next state(m1, b_end)
    :param q: the probability of transition from b_start to b1.
    :param num_states: number of states
    :return: transition matrix according to the ZOOPS model
    """
    # initialization
    transition_matrix = np.zeros(shape=[num_states, num_states], dtype=float)

    # fill probabilities between states according to the TCM model
    transition_matrix[0][1] = p
    transition_matrix[0][0] = 1 - p
    transition_matrix[range(1, num_states-1), range(2, num_states)] = 1
    transition_matrix[-1][0] = 1

    return transition_matrix


def forward_algorithm(seq, transition_matrix, emission_matrix):
    """
    create the forward matrix (in log space) and fill it by using the transition and emission matrices
    :param seq: the sequence
    :param transition_matrix: log space transition matrix
    :param emission_matrix: log space emission matrix
    :return: The forward matrix (the log likelihood value in the cell which represent (b_end, '$'))
    """
    num_states = emission_matrix.shape[0]
    len_seq = len(seq)

    # initialization
    forward_matrix = np.zeros([num_states, len_seq], dtype=float)
    forward_matrix[0][0] = 1

    # moving to log space
    with np.errstate(divide='ignore'):
        forward_matrix = np.log(forward_matrix)

    # fill the table - vectorized version in log space, loop the columns only
    for col_idx in range(1, len_seq):
        last_col = forward_matrix[:, col_idx-1].reshape(-1, 1)
        letter_idx_emission = letters_dict[seq[col_idx]]
        forward_matrix[:, col_idx] = logsumexp(transition_matrix + last_col, axis=0) + emission_matrix[:,
                                                                                       letter_idx_emission]

    return forward_matrix


def backward_algorithm(seq, transition_matrix, emission_matrix):
    """
    create the backward matrix (in log space) and fill it by using the transition and emission matrices
    :param seq: the sequence
    :param transition_matrix: log space transition matrix
    :param emission_matrix: log space emission matrix
    :return: The backward matrix (the log likelihood value in the cell which represent (b_start, '^'))
    """
    num_states = emission_matrix.shape[0]
    len_seq = len(seq)

    # initialization
    backward_matrix = np.zeros([num_states, len_seq], dtype=float)
    backward_matrix[1][-1] = 1

    # moving to log space
    with np.errstate(divide='ignore'):
        backward_matrix = np.log(backward_matrix)

    # fill the table - vectorized version in log space, loop the columns only
    for col_idx in range(len(seq) - 1, 0, -1):
        last_col = backward_matrix[:, col_idx].reshape(-1, 1)
        emission = emission_matrix[:, letters_dict[seq[col_idx]]].reshape(-1, 1)
        backward_matrix[:, col_idx - 1] = logsumexp(transition_matrix.T + emission + last_col, axis=0)
    return backward_matrix


def update_p(N_k_l):
    """
    Update p and q as a part of maximization in order to update the transition matrix later
    :param N_k_l: sufficient statistics for number of times moving for state k to l
    :return: p and q
    """
    log_count_other_transitions_p = logsumexp(N_k_l[0, :])
    p = np.exp(N_k_l[0][1] - log_count_other_transitions_p)
    return p


def update_transition_matrix(emission_matrix, p):
    """
    Update the transition matrix as a part of the maximization
    :param emission_matrix: log scale emission matrix
    :return: the new log scaled transition matrix
    """
    # create new transition matrix by using the new values for p and q
    tran_mat = create_transition_matrix(p, emission_matrix.shape[0])
    with np.errstate(divide='ignore'):
        transition_matrix = np.log(tran_mat)
    return transition_matrix


def baum_welch(sequences, emission_matrix, transition_matrix, convergence_threshold):
    """
    Run the EM algorithm
    :param sequences: list of the sequences
    :param emission_matrix: log scale emission matrix
    :param transition_matrix: log scale transition matrix
    :param convergence_threshold: the convergence threshold
    :return:
    """
    with np.errstate(divide='ignore'):
        init_val = np.log(0)

    ll_hist = []
    prev_ll = None

    while True:
        # init sufficient statistics matrices
        N_k_l = np.full((transition_matrix.shape[0], transition_matrix.shape[1]), init_val)
        N_k_x = np.full((emission_matrix.shape[0], emission_matrix.shape[1]), init_val)

        curr_ll = 0
        for seq in sequences:
            forward_matrix = forward_algorithm(seq, transition_matrix, emission_matrix)
            backward_matrix = backward_algorithm(seq, transition_matrix, emission_matrix)
            ll = forward_matrix[1][-1]
            curr_ll += ll

            # expectation step
            for i in range(len(seq)):
                # letter_idx between 0 and 5 where 0 means '^' and 5 means '$'
                letter_idx = letters_dict[seq[i]]
                # calc N_k_x
                # if 1 <= letter_idx <= 4:
                N_k_x[:, letter_idx] = np.logaddexp(N_k_x[:, letter_idx], forward_matrix[:, i] +
                                                      backward_matrix[:, i] - ll)
                # calc N_k_l
                N_k_l = np.logaddexp(N_k_l, (forward_matrix[:, i - 1].reshape(-1, 1) + transition_matrix +
                                             emission_matrix[:, letter_idx].reshape(1, -1) +
                                             backward_matrix[:, i].reshape(1, -1) - ll))

        # maximization step
        sum_N_k_y = np.array(logsumexp(N_k_x, axis=1)).reshape(1, -1)
        emission_matrix[:, :] = N_k_x - sum_N_k_y.T  # update emission

        p = update_p(N_k_l)
        transition_matrix = update_transition_matrix(emission_matrix, p)  # update transition

        ll_hist.append(curr_ll)

        # stop condition
        if prev_ll is not None and (math.fabs(curr_ll - prev_ll) <= convergence_threshold):
            return emission_matrix, transition_matrix, ll_hist, p
        prev_ll = curr_ll


def restore_most_likely_states(row_idx, col_idx, pointer_matrix):
    """
    return the most likely hidden states for the sequence, derived from the pointer matrix
    :param row_idx: row index to start
    :param col_idx: column index to start
    :param pointer_matrix: our backtrace matrix to restore the solution for most likely hidden states
    :return: most likely hidden states string
    """
    res = ""
    while col_idx > 0:
        # if b1 or b2 states
        if pointer_matrix[row_idx][col_idx] in (2, 3):
            res += 'B'
        # for motif states
        elif pointer_matrix[row_idx][col_idx] > 3:
            res += 'M'
        # update where to go next
        row_idx = pointer_matrix[row_idx][col_idx]
        col_idx -= 1

    res = res[::-1]
    return res


def viterbi(seq, transitions_matrix, emission_matrix):
    """
    find the most likely hidden states that emitted the sequence
    :param seq: the sequence
    :param transitions_matrix: the transition matrix in log space
    :param emission_matrix: the emission matrix in log space
    :return: the most likely hidden states that emitted the sequence
    """
    # initialization
    len_seq = len(seq)
    num_states = emission_matrix.shape[0]
    v_matrix = np.zeros([num_states, len_seq], dtype=float)
    pointer_matrix = np.zeros([num_states, len_seq], dtype=int)

    pointer_matrix[0][0] = 1
    v_matrix[0][0] = 1

    with np.errstate(divide='ignore'):
        v_matrix = np.log(v_matrix)

    # fill the viterbi and pointer matrices in log space by vectorized version- iterate only on columns
    for col_idx in range(1, len_seq):
        last_col = v_matrix[:, col_idx-1].reshape(-1, 1)
        letter_idx_emission = letters_dict[seq[col_idx]]
        emission = emission_matrix[:, letter_idx_emission]

        max_val = np.max(last_col + transitions_matrix, axis=0)
        arg_max_idx = np.argmax(last_col + transitions_matrix, axis=0)

        v_matrix[:, col_idx] = emission + max_val
        pointer_matrix[:, col_idx] = arg_max_idx

    return restore_most_likely_states(1, len_seq-1, pointer_matrix)


def posterior(forward_matrix, backward_matrix, seq):
    """
    calculate the posterior - the most likely hidden state for each index in the given sequence, using the
    forward and backward matrices
    :param forward_matrix: the forward matrix
    :param backward_matrix: the backward matrix
    :param seq: the sequence
    :return: the most likely hidden state for each index in the given sequence s.t 'B' represent the states B1 or B2
    and 'M' represent states of motif.
    """
    # initial the result - empty string
    res = ""

    posterior_matrix = forward_matrix + backward_matrix
    for col_idx in range(1, len(seq)-1):
        # find the index of hidden state which is most likely
        curr_idx = np.argmax(posterior_matrix[:, col_idx])
        if curr_idx >= 4:
            res += 'M'
        else:
            res += 'B'
    return res