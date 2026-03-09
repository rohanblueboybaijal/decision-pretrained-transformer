import gym
import numpy as np

from envs.base_env import BaseEnv


class JunctionEnv(BaseEnv):
    """
    Plus-shaped junction maze: Navigate from the bottom of a corridor up to
    a junction, then choose left, right, or up to reach a hidden goal.

    Actions: 0=right, 1=left, 2=down, 3=up, 4=stay
    Reward: 1 when at goal, 0 otherwise.

    With corridor_length L:
        - Start: (L, 0)
        - Junction: (L, L)
        - Goals: (0, L) left, (2L, L) right, (L, 2L) up
    """

    def __init__(self, corridor_length, goal, horizon):
        self.corridor_length = corridor_length
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        L = corridor_length
        self._valid_cells = self._build_valid_cells()
        self._junction = np.array([L, L], dtype=float)
        self._start = np.array([L, 0], dtype=float)
        grid_max = 2 * L
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_max, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)

    def _build_valid_cells(self):
        L = self.corridor_length
        cells = set()
        for y in range(L + 1):
            cells.add((L, y))
        for x in range(2 * L + 1):
            cells.add((x, L))
        for y in range(L + 1, 2 * L + 1):
            cells.add((L, y))
        return cells

    def sample_state(self):
        cells = list(self._valid_cells)
        idx = np.random.randint(0, len(cells))
        return np.array(cells[idx], dtype=float)

    def sample_action(self):
        action = np.zeros(self.action_space.n)
        action[np.random.randint(0, self.action_space.n)] = 1
        return action

    def reset(self):
        self.current_step = 0
        self.state = self._start.copy()
        return self.state.copy()

    def transit(self, state, action):
        action_idx = np.argmax(action)
        state = np.array(state, dtype=float)

        new_state = state.copy()
        if action_idx == 0:
            new_state[0] += 1
        elif action_idx == 1:
            new_state[0] -= 1
        elif action_idx == 2:
            new_state[1] += 1
        elif action_idx == 3:
            new_state[1] -= 1

        if (int(round(new_state[0])), int(round(new_state[1]))) in self._valid_cells:
            state = new_state

        reward = float(np.all(np.abs(state - self.goal) < 1e-5))
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, reward = self.transit(self.state, action)
        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state.copy(), reward, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        """Privileged expert that knows the goal and navigates optimally."""
        x, y = state[0], state[1]
        gx, gy = self.goal[0], self.goal[1]
        L = self.corridor_length

        if np.abs(x - gx) < 1e-5 and np.abs(y - gy) < 1e-5:
            action_idx = 4  # at goal, stay
        elif np.abs(x - L) < 1e-5 and y < L - 1e-5:
            action_idx = 2  # on bottom corridor, +y toward junction
        elif np.abs(x - L) < 1e-5 and np.abs(y - L) < 1e-5:
            # at junction, move toward goal's corridor
            if gx < L:
                action_idx = 1  # left
            elif gx > L:
                action_idx = 0  # right
            else:
                action_idx = 2  # +y toward upper corridor
        elif np.abs(y - L) < 1e-5 and x < L - 1e-5:
            # on left corridor
            if gx < L and np.abs(gy - L) < 1e-5:
                if x > gx + 1e-5:
                    action_idx = 1  # left toward goal
                else:
                    action_idx = 4  # at goal
            else:
                action_idx = 0  # right, back toward junction
        elif np.abs(y - L) < 1e-5 and x > L + 1e-5:
            # on right corridor
            if gx > L and np.abs(gy - L) < 1e-5:
                if x < gx - 1e-5:
                    action_idx = 0  # right toward goal
                else:
                    action_idx = 4  # at goal
            else:
                action_idx = 1  # left, back toward junction
        elif np.abs(x - L) < 1e-5 and y > L + 1e-5:
            # on upper corridor
            if np.abs(gx - L) < 1e-5 and gy > L:
                if y < gy - 1e-5:
                    action_idx = 2  # +y toward goal
                else:
                    action_idx = 4  # at goal
            else:
                action_idx = 3  # -y back toward junction
        else:
            action_idx = 4  # fallback stay

        action = np.zeros(self.action_space.n)
        action[action_idx] = 1
        return action


class JunctionEnvVec(BaseEnv):
    """Vectorized Junction environment for parallel execution."""

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self._goals = np.array([env.goal for env in envs])
        self._corridor_length = envs[0].corridor_length
        self._valid_cells = envs[0]._valid_cells
        self.horizon = envs[0].horizon
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def sample_state(self):
        return np.array([env.sample_state() for env in self._envs])

    def sample_action(self):
        actions = np.zeros((self._num_envs, self.action_dim))
        actions[np.arange(self._num_envs), np.random.randint(0, self.action_dim, self._num_envs)] = 1
        return actions

    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        L = self._corridor_length
        self.states = np.tile(np.array([L, 0], dtype=float), (self._num_envs, 1))
        return self.states.copy()

    def transit(self, states, actions):
        action_idxs = np.argmax(actions, axis=1)

        new_states = states.copy()
        new_states[:, 0] += (action_idxs == 0).astype(float)
        new_states[:, 0] -= (action_idxs == 1).astype(float)
        new_states[:, 1] += (action_idxs == 2).astype(float)
        new_states[:, 1] -= (action_idxs == 3).astype(float)

        for i in range(self._num_envs):
            cell = (int(round(new_states[i, 0])), int(round(new_states[i, 1])))
            if cell not in self._valid_cells:
                new_states[i] = states[i]

        rewards = (np.linalg.norm(new_states - self._goals, axis=1) < 1e-5).astype(float)
        return new_states, rewards

    def step(self, actions):
        if np.any(self.current_step >= self.horizon):
            raise ValueError("Episode has already ended for some environments")

        self.states, rewards = self.transit(self.states, actions)
        self.current_step += 1
        dones = self.current_step >= self.horizon
        return self.states.copy(), rewards, dones, {}

    def opt_action(self, states):
        actions = np.array([env.opt_action(state) for env, state in zip(self._envs, states)])
        return actions
