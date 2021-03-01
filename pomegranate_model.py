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

start_dist = {x: 0 for x in possible_observations}
start_dist['$'] = 1
start_dist['^'] = 0
start_state = State(DiscreteDistribution(start_dist), 'start')
start_state.distribution.freeze()
end_dist = {x: 0 for x in possible_observations}
end_dist['^'] = 1
end_dist['$'] = 0
end_state = State(DiscreteDistribution(end_dist), 'end')
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
num_of_short_motif_states = 20
short_motif_states = [
    State(emission_motif, name=f'S{i+1}') 
    for i in range(num_of_short_motif_states)
]
longer_motif_states = [
    State(emission_motif, name=f'L{i+1}') 
    for i in range(num_of_short_motif_states + 1)
]


model = HiddenMarkovModel('trial')
model.add_states(short_motif_states)
model.add_states(longer_motif_states)
model.add_states([background_state])
model.add_states([start_state, end_state])

p = 0.01
model.add_transition(model.start, start_state, 1)
model.add_transition(start_state, background_state, 1)
model.add_transition(background_state, background_state, 1-p)
model.add_transition(background_state, short_motif_states[0], 0.75*p)
model.add_transition(background_state, longer_motif_states[0], 0.25*p-0.001)
model.add_transition(background_state, end_state, 0.001)
model.add_transition(end_state, model.end, 1)

for i, Mi in enumerate(short_motif_states[:-1]):
    model.add_transition(Mi, short_motif_states[i + 1], 1-0.75*(1/20))
    model.add_transition(Mi, background_state, 0.75*(1/20))
model.add_transition(short_motif_states[-1], background_state, 1)

for i, Mi in enumerate(longer_motif_states[:-1]):
    model.add_transition(Mi, longer_motif_states[i + 1], 1)
    
q = 0.02
model.add_transition(longer_motif_states[-1], longer_motif_states[-1], q)
model.add_transition(longer_motif_states[-1], background_state, 1-q)
model.bake()

chains = read_chains('pdbtm')[:30]
observation = [np.array(['$'] + list(seq) + ['^']) for seq in get_alpha_sequences(chains)]
# model.fit(observation, algorithm='baum-welch', batches_per_epoch=100, n_jobs=2)
model.fit(observation, algorithm='baum-welch')
track = "".join(state.name for i, state in model.predict(observation[1]))

import json
with open('model_json_all_data_BW2', 'w') as f:
  json.dump(model.to_json(), f)
