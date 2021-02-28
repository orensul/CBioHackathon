import numpy as  np
from pomegranate import (
    DiscreteDistribution,
    HiddenMarkovModel,
    State,
)
from possible_observations import possible_observations
from featch_PDBTM_db import (
    get_seeds,
    read_chains,
    get_alpha_helix_subsequences,
    get_alpha_sequences
)

emission_background = DiscreteDistribution({
    obs: 1/len(possible_observations) for obs in possible_observations
})
# emission_background['^'] = 0
emission_motif = DiscreteDistribution({
    obs: 1/len(possible_observations) for obs in possible_observations
})
# emission_motif['^'] = 0


background_state = State(emission_background, name='background_state')
len_state = State(emission_motif, name='len_state')
num_of_short_motif_states = 10
short_motif_states = [
    State(emission_motif, name=f'MS{i}') for i in range(num_of_short_motif_states)
]
longer_motif_states = [
    State(emission_motif, name=f'ML{i}') for i in range(num_of_short_motif_states + 1)
]


model = HiddenMarkovModel('trial')
model.add_states(short_motif_states)
model.add_states(longer_motif_states)
model.add_states([background_state])

model.add_transition(model.start, background_state, 1)

p = 0.01
model.add_transition(background_state, background_state, 1-p)
model.add_transition(background_state, len_state, p-0.001)

for Mi in short_motif_states:
    model.add_transition(len_state, Mi, 1 / len(short_motif_states) - 0.0001)

model.add_transition(len_state, longer_motif_states[0], num_of_short_motif_states*0.0001)

for i, Mi in enumerate(short_motif_states[:-1]):
    model.add_transition(Mi, short_motif_states[i + 1], 1)

model.add_transition(short_motif_states[-1], background_state, 1)
model.add_transition(background_state, model.end, 0.001)

for i, Mi in enumerate(longer_motif_states[:-1]):
    model.add_transition(Mi, longer_motif_states[i + 1], 1)
q = 0.2
model.add_transition(longer_motif_states[-1], longer_motif_states[-1], q)
model.add_transition(longer_motif_states[-1], background_state, 1-q)

model.bake()

# print(model.edge_count())
# print(model.sample())
chains = read_chains('pdbtm')[:30]
observation = [np.array(list(seq)) for seq in get_alpha_sequences(chains)]
print('START FITTING')
model.fit(observation, algorithm='baum-welch')
print(model.log_probability(observation[0]))
# print(model.viterbi(observation[0]))
print(", ".join(state.name for i, state in model.viterbi(observation[0])[1]))