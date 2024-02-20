from typing import Any, Dict, List

import dm_env
import numpy as np
from dm_env import specs
from gymnasium import spaces
import gymnasium as gym
import sys
sys.modules["gym"] = gym
from math import inf
import copy

import cv2


class Wrapper(dm_env.Environment):
    """Base class for dm_env.Environment wrapper."""

    def __init__(self, env: dm_env.Environment):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(f'Attempted to get missing private attribute {name}')
        return getattr(self.env, name)

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(action)

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def action_spec(self) -> Any:
        return self.env.action_spec()

    def discount_spec(self) -> Any:
        return self.env.discount_spec()

    def observation_spec(self) -> Any:
        return self.env.observation_spec()

    def reward_spec(self) -> Any:
        return self.env.reward_spec()

    def close(self):
        return self.env.close()


class ObservationWrapper(Wrapper):
    """Base class for observation wrapper."""

    def observation_spec(self):
        raise NotImplementedError

    def observation(self, obs: Any) -> Any:
        raise NotImplementedError

    def step(self, action) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.step(action)
        return dm_env.TimeStep(step_type, discount, reward, self.observation(observation))

    def reset(self) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.reset()
        return dm_env.TimeStep(step_type, discount, reward, self.observation(observation))


class RemapObservationWrapper(ObservationWrapper):
    """Select a subset of dictionary observation keys and rename them."""

    def __init__(self, env: dm_env.Environment, mapping: Dict[str, str]):
        super().__init__(env)
        self.mapping = mapping

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return {key: spec[key_orig] for key, key_orig in self.mapping.items()}

    def observation(self, obs):
        assert isinstance(obs, dict)
        return {key: obs[key_orig] for key, key_orig in self.mapping.items()}


class TargetsPositionWrapper(ObservationWrapper):
    """Collects and postporcesses walker/target_rel_{i} relative position vectors into 
    targets_vec (n_targets,2) tensor, and walker/targets_abs_{i} absolute positions 
    into targets_pos tensor."""

    def __init__(self, env: dm_env.Environment, maze_xy_scale, maze_width, maze_height):
        super().__init__(env)
        self.maze_xy_scale = maze_xy_scale
        self.center_ji = np.array([maze_width - 2.0, maze_height - 2.0]) / 2.0

        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert 'walker/target_rel_0' in spec
        assert 'walker/target_abs_0' in spec
        assert 'target_index' in spec

        i = 0
        while f'walker/target_rel_{i}' in spec:
            assert f'walker/target_abs_{i}' in spec
            i += 1

        self.n_targets = i

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        # All targets
        spec['targets_vec'] = specs.Array((self.n_targets, 2), float, 'targets_vec')
        spec['targets_pos'] = specs.Array((self.n_targets, 2), float, 'targets_pos')
        # Current target
        spec['target_vec'] = specs.Array((2,), float, 'target_vec')
        spec['target_pos'] = specs.Array((2,), float, 'target_pos')
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        # All targets
        x_rel = np.zeros((self.n_targets, 2))
        x_abs = np.zeros((self.n_targets, 2))
        for i in range(self.n_targets):
            x_rel[i] = obs[f'walker/target_rel_{i}'][:2] / self.maze_xy_scale
            x_abs[i] = obs[f'walker/target_abs_{i}'][:2] / self.maze_xy_scale + self.center_ji
        obs['targets_vec'] = x_rel
        obs['targets_pos'] = x_abs
        # Current target
        target_ix = int(obs['target_index'])
        obs['target_vec'] = x_rel[target_ix]
        obs['target_pos'] = x_abs[target_ix]
        return obs


class AgentPositionWrapper(ObservationWrapper):
    """Postprocesses absolute_position and absolute_orientation."""

    def __init__(self, env: dm_env.Environment, maze_xy_scale, maze_width, maze_height):
        super().__init__(env)
        self.maze_xy_scale = maze_xy_scale
        self.center_ji = np.array([maze_width - 2.0, maze_height - 2.0]) / 2.0

    def observation_spec(self):
        spec = self.env.observation_spec()
        # absolute_position and absolute_orientation should already be generated by the environment.
        assert isinstance(spec, dict) and 'absolute_position' in spec and 'absolute_orientation' in spec
        # Add agent_pos, measured in grid coordinates
        spec['agent_pos'] = specs.Array((2, ), float, 'agent_pos')
        # Add agent_dir as 2-vector
        spec['agent_dir'] = specs.Array((2, ), float, 'agent_dir')
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        walker_xy = obs['absolute_position'][:2]
        walker_ji = walker_xy / self.maze_xy_scale + self.center_ji
        # agent_pos, measured in grid coordinates, where bottom-left coordinate is (0.1,0.1),
        # and top-right coordinate for a 15x15 maze is (14.9,14.9)
        obs['agent_pos'] = walker_ji
        # Pick orientation vector such, that going forward increases agent_pos in the direction of agent_dir.
        obs['agent_dir'] = obs['absolute_orientation'][:2, 1]
        return obs


class MazeLayoutWrapper(ObservationWrapper):
    """Postprocesses maze_layout observation."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        # maze_layout should already be generated by the environment
        assert isinstance(spec, dict) and 'maze_layout' in spec
        # Change char array to binary array, removing outer walls
        n, m = spec['maze_layout'].shape
        spec['maze_layout'] = specs.BoundedArray((n - 2, m - 2), np.uint8, 0, 1, 'maze_layout')
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        maze = obs['maze_layout']
        maze = maze[1:-1, 1:-1]  # Remove outer walls
        maze = np.flip(maze, 0)  # Flip vertical axis so that bottom-left is at maze[0,0]
        nonwalls = (maze == ' ') | (maze == 'P') | (maze == 'G')
        obs['maze_layout'] = nonwalls.astype(np.uint8)
        return obs


class ImageOnlyObservationWrapper(ObservationWrapper):
    """Select one of the dictionary observation keys as observation."""

    def __init__(self, env: dm_env.Environment, key: str = 'image'):
        super().__init__(env)
        self.key = key

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return spec[self.key]

    def observation(self, obs):
        assert isinstance(obs, dict)
        return obs[self.key]


class DiscreteActionSetWrapper(Wrapper):
    """Change action space from continuous to discrete with given set of action vectors."""

    def __init__(self, env: dm_env.Environment, action_set: List[np.ndarray]):
        super().__init__(env)
        self.action_set = action_set
        self.action_space = spaces.Discrete(len(action_set))

    def action_spec(self):
        return specs.DiscreteArray(len(self.action_set))

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(self.action_set[action])


class TargetColorAsBorderWrapper(ObservationWrapper):
    """MemoryMaze-specific wrapper, which draws target_color as border on the image."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert 'target_color' in spec
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        assert 'target_color' in obs and 'image' in obs
        target_color = obs['target_color']
        img = obs['image']
        B = int(2 * np.sqrt(img.shape[0] // 64))
        img[:, :B] = target_color * 255 * 0.7
        img[:, -B:] = target_color * 255 * 0.7
        img[:B, :] = target_color * 255 * 0.7
        img[-B:, :] = target_color * 255 * 0.7
        return obs


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        subspace1 = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=int)
        subspace2 = spaces.Discrete(6)
        subspace3 = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=int)
        subspace4 = spaces.Box(-inf, inf, (3,), dtype=float)
        self.env.observation_space = spaces.Dict({
            'image': subspace1,
            'prev_action': subspace2,
            'prev_image': subspace3,
            'goal': subspace4
        })
        self.max_episode_steps = 1000
        self.oracle_min_num_actions = 0
        self.actions_for_one_grid = 5

    def reset(self, seed = 0):
        obs, info = self.env.reset(seed)
        self.prev_image = obs["image"]
        new_obs = {"image": obs["image"], "prev_action": 0, "prev_image": self.prev_image, "goal": obs["target_color"]}

        # num of actions to get a target
        self.oracle_min_num_actions = len(obs['path']) * self.actions_for_one_grid

        self.s = 0
        self.reward_sum = 0

        return new_obs, info

    def visualization(self, image, step, reward):
        # 텍스트 정보 설정
        text1 = "Steps: " + str(step)
        text2 = "Rewards: " + str(reward)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (255, 255, 255)
        thickness = 3
        line_type = cv2.LINE_AA

        # 텍스트 크기 계산
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]

        margin = 50

        # 텍스트를 표시할 이미지의 크기 계산
        text1_width, text1_height = text1_size[0], text1_size[1]
        text2_width, text2_height = text2_size[0], text2_size[1]
        image_height, image_width = image.shape[0], image.shape[1]
        combined_image = np.zeros((image_height, image_width + max(text1_width, text2_width) + margin * 2, 3), dtype=np.uint8)

        # 이미지 복사
        combined_image[:, :image_width] = image

        # 텍스트를 합쳐진 이미지의 오른쪽에 추가
        text_x = image_width + margin  # 텍스트와 이미지 사이의 간격
        text1_y = text1_height // 2 + (margin * 2)
        text2_y = text1_height + text2_height // 2 + (margin * 3)
        cv2.putText(combined_image, text1, (text_x, text1_y), font, font_scale, font_color, thickness, line_type)
        cv2.putText(combined_image, text2, (text_x, text2_y), font, font_scale, font_color, thickness, line_type)

        # 결과 이미지 표시
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(1)

    def step(self, action):
        obs, reward, done, truncate, info = self.env.step(action)

        # add num of actions to get a target
        if reward > 0:
            print("YOU RECEIVED THE AMAZING REWARD!!!!!!!!!!!")
            self.oracle_min_num_actions += len(obs['path']) * self.actions_for_one_grid

        self.s += 1
        self.reward_sum += reward
        # self.visualization(cv2.resize(obs["image"], (720, 720), interpolation = cv2.INTER_AREA), self.s, self.reward_sum)

        new_obs = {"image": obs["image"], "prev_action": action, "prev_image": self.prev_image, "goal": obs["target_color"]}
        self.prev_image = copy.deepcopy(obs["image"])

        return new_obs, reward, done, truncate, info

class ReverseModeWrapper(gym.Wrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task

    def reset(self, seed = 0):
        obs, info = self.env.reset(seed)
        self._steps2goal = 0
        self.num_steps = []
        info['steps'] = self.num_steps
        return obs, info

    def step(self, action):
        obs, reward, done, truncate, info = self.env.step(action)
        if self.task.is_color_swapped:
            self.task.target_color_before_swap
            obs['goal'] = self.task.target_color_before_swap

        self._steps2goal += 1
        if reward > 0:
            self.num_steps.append(self._steps2goal)
            self._steps2goal = 0
        
        info['steps'] = self.num_steps
        return obs, reward, done, truncate, info
    
    def test_mode(self):
        self._test = True
    
    def reverse_mode(self, case):
        #print("reverse mode", case)
        self.task.reverse_mode = case
