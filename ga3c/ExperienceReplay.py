import numpy as np
from collections import deque


class ExperienceFrame:
    def __init__(self, frame, action, prev_r):
        self.frame = frame
        self.action = action
        self.reward = np.clip(prev_r, -1, 1)

class ExperienceState:
    def __init__(self, state, action, prev_r):
        self.state = state
        self.action = action
        self.reward = prev_r

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

    def add_experience(self, frame, action, prev_r):
        exp = ExperienceFrame(frame, action, prev_r)
        frame_index = self._top_frame_index + len(self._exps)
        was_full = self.is_full()

        self._exps.append(exp)

        if frame_index >= 3:
            if exp.reward == 0:
                self._zero_reward_indices.append(frame_index)
            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            self._top_frame_index += 1
            cut_frame_index = self._top_frame_index + 3
            if len(self._zero_reward_indices) > 0 and self._zero_reward_indices < cut_frame_index:
                self._zero_reward_indices.popleft()
            if len(self._non_zero_reward_indices) > 0 and self._non_zero_reward_indices < cut_frame_index:
                self._non_zero_reward_indices.popleft()

    def sample_sequence(self):
        """
        Sample 4 successive frames for reward prediction.
        """
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

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
        prev_r = self._exps[raw_start_frame_index + 3].reward
        state = np.array(sampled_frames)
        state = np.transpose(state, [1, 2, 0])
        return (state, action, prev_r)

