from pomegranate import *
from possible_observations import possible_observations
from fetch_pdbtm_db import (
read_chains,
generate_training_data
)


NUM_SHORT_STATES = 20

class CreateModel:
    def __init__(self):
        start_dist = {x: 0 for x in possible_observations}
        start_dist['$'] = 1
        start_dist['^'] = 0
        self.start_state = State(DiscreteDistribution(start_dist), 'start')
        end_dist = {x: 0 for x in possible_observations}
        end_dist['^'] = 1
        end_dist['$'] = 0
        self.end_state = State(DiscreteDistribution(end_dist), 'end')
        self.model = HiddenMarkovModel(name='membraneModel', start=self.start_state, end=self.end_state)
        uniform = {x: 1/len(possible_observations) for x in possible_observations}
        uniform['$'] = 0
        uniform['^'] = 0
        self.uniform = DiscreteDistribution(uniform)
        self.long_states = dict()
        self.short_states = dict()
        self.background_state = None
        self.short_states_keys = None
        self.long_states_keys = None

    def create_short_states(self):
        self.short_states_keys = ['SM' + str(i + 1) for i in range(NUM_SHORT_STATES)]
        for i in range(NUM_SHORT_STATES):
            name_state = 'SM' + str(i+1)
            self.short_states[name_state] = State(self.uniform, name_state)

    def create_long_states(self):
        self.long_states_keys = ['LM' + str(i + 1) for i in range(NUM_SHORT_STATES + 1)]
        for i in range(NUM_SHORT_STATES + 1):
            name_state = 'LM' + str(i+1)
            self.long_states[name_state] = State(self.uniform, name_state)

    def create_background_state(self):
        self.background_state = State(self.uniform, 'B')

    def add_states(self):
        self.model.add_states(self.short_states.values())
        self.model.add_states(self.long_states.values())
        self.model.add_states(self.background_state)

    def create_transitions(self):
        # from start state
        self.model.add_transition(self.model.start, self.background_state, 1)

        # from background states
        self.model.add_transition(self.background_state, self.background_state, 0.25)
        self.model.add_transition(self.background_state, self.short_states[self.short_states_keys[0]], 0.25)
        self.model.add_transition(self.background_state, self.long_states[self.long_states_keys[0]], 0.25)
        self.model.add_transition(self.background_state, self.model.end, 0.25)

        # short states
        for i in range(NUM_SHORT_STATES - 1):
            curr_short_state = self.short_states_keys[i]
            next_short_state = self.short_states_keys[i+1]
            self.model.add_transition(self.short_states[curr_short_state], self.short_states[next_short_state], 0.5)
            self.model.add_transition(self.short_states[curr_short_state], self.background_state, 0.5) # TODO: should we go back from SM1-SM5?
        self.model.add_transition(self.short_states[self.short_states_keys[NUM_SHORT_STATES-1]], self.background_state, 1)

        # long states
        for i in range(NUM_SHORT_STATES):
            curr_long_state = self.long_states_keys[i]
            next_long_state = self.long_states_keys[i+1]
            self.model.add_transition(self.long_states[curr_long_state], self.long_states[next_long_state], 1)
        self.model.add_transition(self.long_states[self.long_states_keys[NUM_SHORT_STATES]], self.background_state, 0.5)
        self.model.add_transition(self.long_states[self.long_states_keys[NUM_SHORT_STATES]], self.long_states[self.long_states_keys[NUM_SHORT_STATES]], 0.5)

    def init_model(self):
        self.create_short_states()
        self.create_long_states()
        self.create_background_state()
        self.create_transitions()
        self.model.bake()

m = CreateModel()
m.init_model()

state_names = ['B', 'start', 'end']
state_names.extend(m.short_states_keys)
state_names.extend(m.long_states_keys)

chains = read_chains('pdbtm')
sequence_train, label_train = generate_training_data(chains)
sequence_test = sequence_train[101]
sequence_train = sequence_train[:100]
label_test = label_train[101]
label_train = label_train[:100]
m.model.fit(sequences=sequence_train,  labels=label_train, algorithm='labeled')
print(m.model.viterbi(sequence_test))


# sequence_test = [['$', 'A', 'D', 'C', 'A', 'Y', 'G', 'A', 'A', 'G', 'D', '^'], ['$', 'A', 'D', 'D', 'D', 'A', '^']]
# label_test = [['start','B', 'B', 'B', 'SM1', 'SM2', 'SM3', 'SM4', 'B', 'B', 'B', 'end'], ['start','B', 'SM1', 'SM2', 'B', 'B', 'end']]
# m.model.fit(sequences=sequence_test,  labels=label_test, algorithm='labeled')
# # m.model.log_probability(['$','A', 'D', 'D', 'D', 'A', '^'])
# print(m.model.viterbi(['$', 'A', 'D', 'C', 'A', 'Y', 'G', 'A', 'A', 'G', 'D', '^']))

