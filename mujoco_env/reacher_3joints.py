import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Optional
import mujoco


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReacherEnv_3joints(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "reacher_3joints.xml",
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:3]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[3:],
                self.data.qvel.flat[:3],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )

    def reset_specific(
        self,
        state,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_specific_model(state)
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def reset_specific_model(self, state):
        qpos = np.zeros(5)
        qvel = np.zeros(5)
        
        #reset qpos
        qpos[3:] = state[6:8]
        '''
        if state[2] > 0: 
            theta0 = np.arccos(state[0])
        else: 
            theta0 = 2 * math.pi - np.arccos(state[0])
            
        if state[3] > 0: 
            theta1 = np.arccos(state[1])
        else: 
            theta1 = 2 * math.pi - np.arccos(state[1])
        '''
        theta0 = np.arctan2(state[3], state[0])
        theta1 = np.arctan2(state[4], state[1])
        theta2 = np.arctan2(state[5], state[2])
        #print("cos")
        #print(state[0])
        #print(np.cos(theta0))
        qpos[0] = theta0
        qpos[1] = theta1
        qpos[2] = theta2
        
        #reset qvel
        qvel[:3] = state[8:11]
        qvel[-2:] = 0
        
        #reset position_fingertip - position_target
        self.data.body("target").xpos = np.append(qpos[3:], 0.01)
        self.data.body("fingertip").xpos = state[-3:] + self.data.body("target").xpos
        
        self.set_specific_state(qpos, qvel)
        return self._get_obs()
        
    def set_specific_state(self, qpos, qvel):
        """Set the joints position qpos and velocity qvel of the model.

        Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
