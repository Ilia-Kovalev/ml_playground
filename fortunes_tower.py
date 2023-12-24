
from argparse import ArgumentParser
import traceback

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space

import rl.core
import rl.memory
import rl.agents
import rl.policy
import keras


class _Deck:
    N_CARD_TYPES = 8

    def __init__(self, type='Emerald'):
        n_numbered_cards = {
            'Emerald': 10,
            'Ruby': 9,
            'Diamond': 8,
        }[type]

        self._cards = sum(
            ([i] * n_numbered_cards for i in range(1, self.N_CARD_TYPES)), 4*[0])
        self._i_card = 0
        self.cards_set = [4] + (self.N_CARD_TYPES - 1) * [n_numbered_cards]
        self.disable_shuffling = False

    def shuffle(self, np_random=None):
        if np_random is None:
            np_random = np.random.Generator(np.random.PCG64())

        if not self.disable_shuffling:
            np_random.shuffle(self._cards)

        self._i_card = 0

    def draw(self):
        c = self._cards[self._i_card]
        self._i_card += 1
        return c

    def move_sequence_forward(self, sequence):
        for i, c in enumerate(sequence):
            i_c = self.remaining_cards[i:].index(c) + self._i_card + i
            self._cards[i_c], self._cards[self._i_card +
                                          i] = self._cards[self._i_card + i], self._cards[i_c]

    def exchange_drown_card(self, card_to_return, replacement):
        i_c = self._cards[:self._i_card].index(card_to_return)
        i_r = self.remaining_cards.index(replacement) + self._i_card
        self._cards[i_c], self._cards[i_r] = self._cards[i_r], self._cards[i_c]

    @property
    def remaining_cards(self):
        return self._cards[self._i_card:]


class Game(gym.Env):

    metadata = {
        "render_modes": ["ansi"]
    }
    render_mode = "ansi"

    N_ROWS = 8
    MAX_ROW_SIZE = N_ROWS

    def __init__(self, deck_type='Emerald',
                 n_rounds=1,
                 disable_jackpot=False,
                 training_reward=False):
        self.action_space = spaces.Discrete(2)
        self._n_rounds = n_rounds
        self._disable_jackpot = disable_jackpot
        self._training_reward = training_reward

        self._rows = self.N_ROWS * [[]]

        max_reward = self.MAX_ROW_SIZE * \
            sum(i * (i + 1) for i in range(1, self.N_ROWS - 1))
        self.reward_range = (-15., max_reward)

        self._deck = _Deck(deck_type)

        self.observation_space = spaces.Dict({
            'multiplier': spaces.Discrete(1),
            'gate_closed': spaces.Discrete(1),
            'remaining_cards': spaces.MultiDiscrete([1] * _Deck.N_CARD_TYPES),
            'current_row': spaces.Tuple(spaces.MultiBinary(_Deck.N_CARD_TYPES)
                                        for _ in range(self.MAX_ROW_SIZE))
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._i_round = 0
        self._new_round()

        return self._observed_state

    def _new_round(self):
        self._deck.shuffle(self.np_random)

        self._rows = 8 * [[]]
        self._rows[0] = [self._deck.draw()]
        self._rows[1] = [self._deck.draw(), self._deck.draw()]

        self._i_row = 2

        if self._rows[1][0] == self._rows[1][1]:
            self._multiplier = 2
        else:
            self._multiplier = 1

        self._prev_reward = self.reward_if_stop
        self._reward_given = 15

    def step(self, action):
        def stop():
            self._i_round += 1
            reward = 0 if self._training_reward else self.reward_if_stop - 15

            res = self._observed_state, reward, self._game_over, {}
            self._new_round()
            return res

        def bust():
            self._i_round += 1
            reward = -15

            out = self._observed_state, reward, self._game_over, {}
            self._new_round()
            return out

        def success():
            if self._i_row == self.N_ROWS:
                return stop()

            reward = self.reward_if_stop - self._prev_reward if self._training_reward else 0

            self._prev_reward = self.reward_if_stop
            self._reward_given += reward

            return self._observed_state, reward, False, {}

        if not action:
            return stop()

        pr = self._current_row
        r = [self._deck.draw() for _ in range(self._i_row + 1)]
        self._rows[self._i_row] = r

        self._i_row += 1

        if 0 in r:
            return success()

        if r.count(r[0]) == len(r):
            self.multiplier = len(r)
            return success()

        for i in range(len(pr)):
            for j in range(i, i + 2):
                if r[j] == pr[i]:
                    if not self._rows[0] or self._rows[0][0] == r[j] or i and self._rows[0][0] == pr[i-1]:
                        return bust()
                    r[j] = self._rows[0][0]
                    self._rows[0] = []
                    if r[j] == 0:
                        return success()

        return success()

    def render(self, mode='ansi'):
        assert mode == 'ansi'

        out = '[#]' if self._rows[0] else '[ ]'
        return '\n'.join([out] + [str(r) for r in self._rows[1:] if r])

    @property
    def gate_closed(self):
        return self._rows[0]

    @property
    def _game_over(self):
        return self._i_round >= self._n_rounds

    @property
    def deck(self):
        return self._deck

    @property
    def reward_if_stop(self):
        if self._i_row == self.N_ROWS and not self._disable_jackpot:
            return self._multiplier * sum(sum(r) for r in self._rows[1:])
        else:
            return self._multiplier * sum(self._current_row)

    @property
    def _current_row(self):
        return self._rows[self._i_row - 1]

    @property
    def _observed_state(self):
        rc = self._deck.remaining_cards
        if self._rows[0]:
            rc.append(self._rows[0][0])

        er = [np.zeros((_Deck.N_CARD_TYPES,), np.int8)
              for i in range(self.MAX_ROW_SIZE)]
        for i, c in enumerate(self._current_row):
            er[i][c] = 1

        return {
            'multiplier': self._multiplier,
            'gate_closed': len(self._rows[0]),
            'remaining_cards': [rc.count(i) for i in range(_Deck.N_CARD_TYPES)],
            'current_row': np.concatenate(er),
        }


class GameWithCustomInput(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_rounds=1, **kwargs)

    def reset(self, row):
        assert len(row) == 2

        self._deck.disable_shuffling = True
        self._deck.shuffle()
        self._deck.move_sequence_forward([0] + row)
        return super().reset()

    def step(self, action, new_row, gate):
        assert len(new_row) == self._i_row + 1

        if self.gate_closed:
            for c in range(_Deck.N_CARD_TYPES):
                if c == self._rows[0][0] and new_row.count(c) == self._deck.remaining_cards.count(c) + 1:
                    self._deck.exchange_drown_card(
                        self._rows[0][0], self._deck.remaining_cards[-1])

        if not gate is None and gate != self._rows[0][0]:
            self._deck.exchange_drown_card(self._rows[0][0], gate)
            self._rows[0][0] = gate

        self._deck.move_sequence_forward(new_row)

        return super().step(action)


class Processor(rl.core.Processor):
    def __init__(self, env):
        super().__init__()
        self._obs_shape = flatten_space(env.observation_space).shape
        self._normalizer = 1 / np.array([
            [env.MAX_ROW_SIZE, 1]
            + env.deck.cards_set
            + [1] *
            flatten_space(env.observation_space['current_row']).shape[0]
        ], dtype=np.float32)
        self._normalizer = np.squeeze(self._normalizer)

    def process_observation(self, observation):
        obs = np.zeros(shape=self._obs_shape, dtype=np.float32)
        obs[0] = observation['multiplier']
        obs[1] = observation['gate_closed']
        obs[2:2+_Deck.N_CARD_TYPES] = observation['remaining_cards']
        obs[2 +
            _Deck.N_CARD_TYPES:] = np.array(observation['current_row']).flatten()
        return obs * self._normalizer

    # def process_state_batch(self, batch):
    #     return batch.reshape((1, -1))
        # if len(batch.shape) > 1:
        #     return np.squeeze(batch, axis=0)
        # return np.reshape(batch, (1, -1))


class Adviser:
    def __init__(self, agent: rl.core.Agent, env: GameWithCustomInput):
        self._agent = agent
        self._env = env
        self._action = 0
        self.reset()

    def reset(self):
        while True:
            try:
                print('\n\nNew round\n')
                cards = input('Enter 2 cards: ')
                self._observation = self._env.reset([int(c) for c in cards])
                self._action = self._agent.forward(
                    self._agent.processor.process_observation(self._observation))
                break
            except KeyboardInterrupt:
                return
            except:
                print(traceback.format_exc(), '\n\n')

    def step(self):
        while True:
            print('Advised acton: ', self._action)
            try:
                cards = input(
                    f'Enter {self._env._i_row + 1} cards of nothing to reset: ')
                if not cards:
                    self.reset()
                    continue

                assert len(cards) == self._env._i_row + 1
                cards = [int(c) for c in cards]

                gate = self._env.gate_closed
                if self._env.gate_closed:
                    gate = input(
                        f'Enter exposed gate card or enter nothing if gate card was not exposed: ')
                    if gate:
                        gate = int(gate)
                    else:
                        gate = None

                self._observation, reward, terminated, _ = self._env.step(
                    self._action, cards, gate)

                print('Reward for step: ', reward)
                print('Reward if hold ', self._env.reward_if_stop)

                if terminated:
                    self.reset()
                    continue

                print('Game state:\n', self._env.render(), '\n')

                self._action = self._agent.forward(
                    self._agent.processor.process_observation(self._observation))
            except KeyboardInterrupt:
                return
            except:
                print(traceback.format_exc(), '\n\n')


def create_agent(env):
    nb_actions = env.action_space.n

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1,) + flatten_space(env.observation_space).shape))
    model.add(keras.layers.Dense(140))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(70))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(30))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(nb_actions))
    model.add(keras.layers.Activation('softmax'))

    processor = Processor(env)

    # agt = rl.agents.SARSAAgent(
    #     model=model,
    #     processor=processor,
    #     nb_actions=nb_actions,
    #     train_interval=10,
    # )
    # agt.compile(keras.optimizers.Adam())

    agt = rl.agents.DQNAgent(
        model=model,
        processor=processor,
        memory=rl.memory.SequentialMemory(limit=50000, window_length=1),
        batch_size=50,
        nb_actions=nb_actions,
        nb_steps_warmup=10,
        train_interval=1000,
        memory_interval=2,
        policy=rl.policy.BoltzmannQPolicy(),
        enable_double_dqn=True,
        enable_dueling_network=True,
    )
    agt.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    return agt


def _consult():
    deck_type = input(
        'Enter deck type ("Emerald", "Ruby", "Diamond", or nothing for "Emerald"): ')
    env = GameWithCustomInput(deck_type if deck_type else 'Emerald')
    agt = create_agent(env)
    agt.load_weights('fortune_tower_params.h5f')

    agt.reset_states()

    adv = Adviser(agt, env)
    adv.step()


def _train():
    env = Game('Emerald', n_rounds=5000,
               disable_jackpot=False, training_reward=True)

    agt = create_agent(env)

    try:
        agt.load_weights('fortune_tower_params.h5f')
    except:
        pass

    agt.fit(env,
            nb_steps=1000000,
            visualize=False,
            verbose=2,
            log_interval=None)
    agt.save_weights('fortune_tower_params.h5f', overwrite=True)


def _test():
    N = 10000
    env = Game(n_rounds=N, training_reward=False)
    env.reset(seed=1)

    agt = create_agent(env)
    agt.load_weights('fortune_tower_params.h5f')
    agt.reset_states()

    total_reward = 0
    obs = env.reset()
    finished = False
    total_act = 0
    total_steps = 0
    while not finished:
        # print(env.render())
        act = agt.forward(agt.processor.process_observation(obs))
        total_act += act
        total_steps += 1
        obs, reward, finished, _ = env.step(act)
        total_reward += reward
        # print(f'Got reward: {reward}\n')

    print(f'Average reward {total_reward / N}')
    print(f'Average action {total_act / total_steps}')


def _main():
    parser = ArgumentParser()
    parser.add_argument('operation', default='consult',
                        choices=('consult', 'train', 'test'))

    args = parser.parse_args()

    return {
        'consult': _consult,
        'train': _train,
        'test': _test,
    }[args.operation]()


if __name__ == '__main__':
    _main()
