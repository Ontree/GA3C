import numpy as np
from collections import deque


class ExperienceFrame:
    def __init__(self, frame, action, reward):
        self.frame = frame
        self.action = action
        if reward == None:
            return
        if reward == 0:
            self.reward = 1.0
        elif reward > 0:
            self.reward = 2.0
        else:
            self.reward = .0


class ExperienceReplay:
    def __init__(self, history_size):
        self._history_size = history_size
        self._exps = deque(maxlen=history_size)
        # frame indices for zero rewards
        self._zero_reward_indices = deque()
        # frame indices for non zero rewards
        self._non_zero_reward_indices = deque()
        self._top_frame_index = 0

    def is_full(self):
        return len(self._exps) >= self._history_size

    def add_experience(self, frame, action, reward):
        exp = ExperienceFrame(frame, action, reward)
        frame_index = self._top_frame_index + len(self._exps)
        was_full = self.is_full()

        self._exps.append(exp)
        if reward != None:
            if reward == 0:
                self._zero_reward_indices.append(frame_index)
            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            self._top_frame_index += 1
            while len(self._zero_reward_indices) > 0 and self._zero_reward_indices[0] < self._top_frame_index:
                self._zero_reward_indices.popleft()
            while len(self._non_zero_reward_indices) > 0 and self._non_zero_reward_indices[0] < self._top_frame_index:
                self._non_zero_reward_indices.popleft()

    def sample_sequence(self):
        """
        Sample 4 successive frames for reward prediction.
        """
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self._zero_reward_indices) == 0 and len(self._non_zero_reward_indices) == 0:
            raise RuntimeError("no valid frames")

        if len(self._zero_reward_indices) == 0:
            # zero rewards container was empty
            from_zero = False
        elif len(self._non_zero_reward_indices) == 0:
            # non zero rewards container was empty
            from_zero = True

        if from_zero:
            index = np.random.randint(len(self._zero_reward_indices))
            end_frame_index = self._zero_reward_indices[index]
        else:
            index = np.random.randint(len(self._non_zero_reward_indices))
            end_frame_index = self._non_zero_reward_indices[index]

        start_frame_index = end_frame_index - 3
        raw_start_frame_index = start_frame_index - self._top_frame_index


        sampled_frames = []

        for i in range(4):
            frame = self._exps[raw_start_frame_index + i].frame
            sampled_frames.append(frame)

        action = self._exps[raw_start_frame_index + 3].action
        reward = self._exps[raw_start_frame_index + 3].reward
        state = np.array(sampled_frames)
        state = np.transpose(state, [1, 2, 0])
        return (state, reward, action)

