from pomegranate import (
    DiscreteDistribution,
    HiddenMarkovModel,
    State,
)
from model.config import possible_observations
from model.config import num_motif_states
from data.fetch_pdbtm_db import get_alpha_helix_subseq_len_dist, get_alpha_helix_subsequences, read_chains


def create_model():
    length_dist = get_alpha_helix_subseq_len_dist(get_alpha_helix_subsequences(read_chains('../data/pdbtm')))

    start_dist = {x: 0 for x in possible_observations}
    start_dist['$'] = 1
    start_dist['^'] = 0
    start_state = State(DiscreteDistribution(start_dist), 'None-start')
    start_state.distribution.freeze()
    end_dist = {x: 0 for x in possible_observations}
    end_dist['^'] = 1
    end_dist['$'] = 0
    end_state = State(DiscreteDistribution(end_dist), 'None-end')
    end_state.distribution.freeze()

    emission_background = DiscreteDistribution({
      **{obs: 1/len(possible_observations) for obs in possible_observations},
      **{'$': 0, '^': 0}
    })
    emission_motif = DiscreteDistribution({
      **{obs: 1/len(possible_observations) for obs in possible_observations},
      **{'$': 0, '^': 0}
    })


    background_state = State(emission_background, name='B')
    num_of_short_motif_states = num_motif_states
    short_motif_states = [
      State(emission_motif, name=f'SM{i+1}')
      for i in range(num_of_short_motif_states)
    ]
    longer_motif_states = [
      State(emission_motif, name=f'LM{i+1}')
      for i in range(num_of_short_motif_states + 1)
    ]

    model = HiddenMarkovModel('trial')
    model.add_states(short_motif_states)
    model.add_states(longer_motif_states)
    model.add_states([background_state])
    model.add_states([start_state, end_state])

    p = 0.01
    prob_len_lower_than_threshold = sum(length_dist[:num_motif_states])
    model.add_transition(model.start, start_state, 1)
    model.add_transition(start_state, background_state, 1)
    model.add_transition(background_state, background_state, 1-p)
    model.add_transition(background_state, short_motif_states[0], prob_len_lower_than_threshold*p)
    model.add_transition(background_state, longer_motif_states[0], (1-prob_len_lower_than_threshold)*p-0.001)
    model.add_transition(background_state, end_state, 0.001)
    model.add_transition(end_state, model.end, 1)

    for i, Mi in enumerate(short_motif_states[:-1]):
      temp_p = length_dist[i]
      model.add_transition(Mi, short_motif_states[i + 1], 1 - temp_p)
      model.add_transition(Mi, background_state, temp_p)
    model.add_transition(short_motif_states[-1], background_state, 1)

    for i, Mi in enumerate(longer_motif_states[:-1]):
      model.add_transition(Mi, longer_motif_states[i + 1], 1)

    q = 0.02
    model.add_transition(longer_motif_states[-1], longer_motif_states[-1], q)
    model.add_transition(longer_motif_states[-1], background_state, 1-q)
    model.bake()
    return model
