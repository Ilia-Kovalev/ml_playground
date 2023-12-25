from argparse import ArgumentParser
import traceback

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, NormalizeObservation

import tianshou as ts
from tianshou.utils.net.common import Net
import torch
from torch.utils.tensorboard import SummaryWriter


class _Deck:
    N_CARD_TYPES = 8

    def __init__(self, type='Ruby'):
        n_numbered_cards = {
            'Ruby': 10,
            'Emerald': 9,
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

    def __init__(self, deck_type='Ruby'):
        self.action_space = spaces.Discrete(2)

        self._rows = self.N_ROWS * [[]]

        self._deck = _Deck(deck_type)

        self.reward_range = (-15, self.MAX_ROW_SIZE *
                             sum(i * i + 1 for i in range(self.MAX_ROW_SIZE)))

        self.observation_space = spaces.Dict({
            'multiplier': spaces.Box(0, 8, dtype=int),
            'gate_closed': spaces.Box(0, 1, dtype=int),
            'remaining_cards': spaces.Tuple(spaces.Box(0, c, dtype=int) for c in self._deck.cards_set),
            'current_row': spaces.Tuple(spaces.MultiBinary(_Deck.N_CARD_TYPES)
                                        for _ in range(self.MAX_ROW_SIZE)),
            'current_prize': spaces.Box(*self.reward_range, dtype=int),
            'turn': spaces.Box(0, self.N_ROWS - 2, dtype=int),
            'chance_to_bust': spaces.Box(0, 1),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._deck.shuffle(self.np_random)

        self._rows = 8 * [[]]
        self._rows[0] = [self._deck.draw()]
        self._rows[1] = [self._deck.draw(), self._deck.draw()]

        self._i_row = 2

        if self._rows[1][0] == self._rows[1][1]:
            self._multiplier = 2
        else:
            self._multiplier = 1

        return self._observed_state, {}

    def step(self, action):
        def stop():
            return self._observed_state, self.reward_if_stop - 15, True, False, {}

        def bust():
            return self._observed_state, -15, True, False, {}

        def success():
            if self._i_row == self.N_ROWS:
                return stop()

            return self._observed_state, 0, False, False, {}

        if not action:
            return stop()

        pr = self._current_row
        r = [self._deck.draw() for _ in range(self._i_row + 1)]
        self._rows[self._i_row] = r

        self._i_row += 1

        if 0 in r:
            return success()

        if r.count(r[0]) == len(r):
            self._multiplier = len(r)
            return success()

        for i in range(len(pr)):
            for j in range(i, i + 2):
                if r[j] == pr[i]:
                    if not self._rows[0] or self._rows[0][0] == r[j] or i and self._rows[0][0] == pr[i]:
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
    def i_row(self):
        return self._i_row

    @property
    def gate_closed(self):
        return self._rows[0]

    @property
    def deck(self):
        return self._deck

    @property
    def reward_if_stop(self):
        if self._i_row == self.N_ROWS:
            return self._multiplier * sum(sum(r) for r in self._rows[1:])
        else:
            return self._multiplier * sum(self._current_row)

    @property
    def _current_row(self):
        return self._rows[self._i_row - 1]

    @property
    def _chance_of_misfortune(self):
        c = self._deck.remaining_cards
        r = self._current_row

        nc = len(c)
        p_hero = self._i_row * c.count(0) / nc

        res = c.count(r[0]) * bool(r[0]) + c.count(r[-1]) * bool(r[-1])
        for i in range(0, len(r) - 1):
            res += c.count([i]) * bool(r[i]) + c.count(r[i+1]) * bool(r[i+1])

        return (res / nc) * (1 - p_hero)

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
            'multiplier': [self._multiplier],
            'gate_closed': [len(self._rows[0])],
            'remaining_cards': [rc.count(i) for i in range(_Deck.N_CARD_TYPES)],
            'current_row': er,
            'current_prize': self.reward_if_stop,
            'turn': self._i_row - 2,
            'chance_to_bust': self._chance_of_misfortune
        }


class GameWithCustomInput(Game):
    def __init__(self, *args, wrapper=id, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrapped = wrapper(self)

    def c_reset(self, row):
        assert len(row) == 2

        self._deck.disable_shuffling = True
        self._deck.shuffle()
        self._deck.move_sequence_forward([0] + row)
        return self._wrapped.reset()

    def c_step(self, action, new_row, gate):
        if not new_row:
            return self._wrapped.step(action)

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

        return self._wrapped.step(action)


class Adviser:
    def __init__(self, env: GameWithCustomInput, file):
        self._env = env
        self._action = 0

        self._policy = create_policy(env._wrapped)
        self._policy.load_state_dict(torch.load(file))
        self._policy.eval()

        self.reset()

    def reset(self):
        while True:
            try:
                print('\n\nNew round\n')
                cards = input('Enter 2 cards: ')
                self._observation, _ = self._env.c_reset(
                    [int(c) for c in cards])
                self._action = self._policy.forward(ts.data.Batch(
                    obs=np.expand_dims(self._observation, 0), info={}))['act'][0]
                break
            except KeyboardInterrupt:
                return
            except:
                print(traceback.format_exc(), '\n\n')

    def step(self):
        while True:
            print('Game state:\n', self._env.render(), '\n')
            print('Reward if stop:', self._env.reward_if_stop)
            print('Advised acton: ', self._action)
            try:
                cards = input(
                    f'Enter {self._env.i_row + 1} cards of nothing to reset: ')
                if not cards:
                    action = 0

                else:
                    action = 1
                    assert len(cards) == self._env.i_row + 1
                    cards = [int(c) for c in cards]

                    gate = None
                    if self._env.gate_closed:
                        gate = input(
                            f'Enter exposed gate card or enter nothing if gate card was not exposed: ')
                        if gate:
                            gate = int(gate)
                        else:
                            gate = None

                self._observation, reward, terminated, _, _ = self._env.c_step(
                    action, cards, gate)

                if terminated:
                    print('Reward: ', reward)
                    self.reset()
                    continue

                self._action = self._policy.forward(
                    ts.data.Batch(obs=np.expand_dims(self._observation, 0), info={}))['act'][0]
            except KeyboardInterrupt:
                return
            except:
                print(traceback.format_exc(), '\n\n')


def normalize_env(env):
    return NormalizeObservation(FlattenObservation(env))


def create_policy(env):
    net = Net(
        state_shape=env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[10],
    )

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    return ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=.9,
        action_space=env.action_space,
        estimation_step=100,
    )


def _train():
    type = 'Diamond'

    train_envs = ts.env.DummyVectorEnv(
        [lambda: normalize_env(Game(type)) for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv(
        [lambda: normalize_env(Game(type)) for _ in range(10)])

    policy = create_policy(normalize_env(Game(type)))

    try:
        policy.load_state_dict(torch.load('dqn.pth'))
    except:
        pass

    train_collector = ts.data.Collector(
        policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(
        policy, test_envs, exploration_noise=True)

    def save(epoch, env_step, gradient_step):
        n = f'checkpoints/{epoch}_{env_step}_{gradient_step}.pth'
        torch.save(policy.state_dict(), n)
        return n

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=150,
        step_per_epoch=10000,
        step_per_collect=1000,
        episode_per_test=10000,
        batch_size=256,
        stop_fn=lambda mean_rewards: mean_rewards >= 5,
        train_fn=lambda epoch, env_step: policy.set_eps(.2),
        test_fn=lambda epoch, env_step: policy.set_eps(0),
        logger=ts.utils.TensorboardLogger(SummaryWriter('log/dqn')),
        resume_from_log=True,
        save_checkpoint_fn=save,
    ).run()

    print(f'Finished training! Use {result["duration"]}')
    torch.save(policy.state_dict(), 'dqn.pth')


def _test():
    N = 10000
    env = normalize_env(Game('Ruby'))
    env.reset(seed=1)

    policy = create_policy(env)
    policy.load_state_dict(torch.load('dqn_ruby.pth'))
    policy.eval()

    total_reward = 0

    total_act = 0
    total_steps = 0
    total_length = 0
    total_jackpots = 0
    for _ in range(N):
        obs = env.reset()[0]
        finished = False
        length = 0
        while not finished:
            # print(env.render())
            act = policy.forward(ts.data.Batch(
                obs=np.expand_dims(obs, 0), info={}))['act'][0]
            total_act += act
            total_steps += 1
            length += 1
            obs, reward, finished, _, _ = env.step(act)
            total_reward += reward
            if env.unwrapped._i_row == env.unwrapped.N_ROWS and reward > 0:
                total_jackpots += 1
            # print(f'Got reward: {reward}\n')

        total_length += length

    print(f'Average reward {total_reward / N}')
    print(f'Average action {total_act / total_steps}')
    print(f'Average length {total_length / N}')
    print(f'Jackpots {total_jackpots}')


def _consult():
    deck_type = input(
        'Enter deck type ("Ruby", "Emerald", "Diamond", or nothing for "Ruby"): ')
    deck_type = deck_type if deck_type else 'Emerald'

    env = GameWithCustomInput(deck_type, wrapper=normalize_env)

    files = {'Emerald': 'dqn_emerald.pth',
             'Ruby': 'dqn_ruby.pth', 'Diamond': 'dqn_diamond.pth'}

    adv = Adviser(env, files[deck_type])
    adv.step()


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
