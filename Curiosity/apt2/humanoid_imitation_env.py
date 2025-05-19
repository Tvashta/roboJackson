import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from mujoco import mj_name2id, mjtObj
import matplotlib.pyplot as plt

class HumanoidImitationEnv(gym.Wrapper):
    def __init__(self, reference_motion, joint_names, reward_scale=10.0, frame_skip=4):
        super().__init__(gym.make("Humanoid-v5", render_mode=None))
        self.model = self.unwrapped.model
        self.data = self.unwrapped.data
        self.reference_motion = reference_motion
        self.joint_names = joint_names
        self.frame_idx = 0
        self.frame_skip = frame_skip

        # Curriculum variables
        self.reward_scale = reward_scale
        self.imitation_weight = 0.1
        self.imitation_weight_max = reward_scale
        self.imitation_schedule_steps = 1000
        self.episode_idx = 0

        # Fall tracking for stats
        self.fall_threshold = 20
        self.total_episodes = 0
        self.early_fall_count = 0

        # Joint mapping
        self.joint_mapping = {
            "pelvis": "root",
            "left_hip": ["left_hip_x", "left_hip_y", "left_hip_z"],
            "right_hip": ["right_hip_x", "right_hip_y", "right_hip_z"],
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_ankle": "left_foot",
            "right_ankle": "right_foot",
            "left_shoulder": ["left_shoulder1", "left_shoulder2"],
            "right_shoulder": ["right_shoulder1", "right_shoulder2"],
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_wrist": "left_hand",
            "right_wrist": "right_hand",
        }

    def reset(self, seed=None, options=None):
        self.frame_idx = 0
        self.episode_idx += 1
        self._update_curriculum()
        return self.env.reset(seed=seed, options=options)

    def _update_curriculum(self):
        ratio = min(1.0, self.episode_idx / self.imitation_schedule_steps)
        self.imitation_weight = self.imitation_weight_max * ratio

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if self.frame_idx < len(self.reference_motion):
            ref_frame = self.reference_motion[self.frame_idx]
            ik_reward = self.ik_imitation_reward(ref_frame)
            total_reward += ik_reward
            self.frame_idx += 1
        else:
            terminated = True

        if self.is_fallen():
            terminated = True
            total_reward -= 200  # Stronger penalty for falling

        if terminated or truncated:
            self.total_episodes += 1
            if self.frame_idx < self.fall_threshold:
                self.early_fall_count += 1

        return obs, total_reward, terminated, truncated, info

    def is_fallen(self, height_thresh=0.7, tilt_thresh_deg=60):
        try:
            torso_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, "torso")
            torso_height = self.data.xpos[torso_id][2]
            if torso_height < height_thresh:
                return True
            z_axis = self.data.xmat[torso_id].reshape(3, 3)[:, 2]
            tilt_angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, [0, 0, 1]), -1.0, 1.0)))
            return tilt_angle > tilt_thresh_deg
        except mujoco.MujocoError:
            return True

    def compute_upright_reward(self):
        try:
            torso_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, "torso")
            xmat = self.data.xmat[torso_id].reshape(3, 3)
            up = xmat[:, 2]
            uprightness = np.dot(up, np.array([0, 0, 1]))
            return max(0.0, uprightness)
        except mujoco.MujocoError:
            return 0.0

    def compute_com_reward(self):
        try:
            com = self.data.subtree_com[0]
            pelvis_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, "root")
            pelvis_pos = self.data.xpos[pelvis_id]
            offset = np.linalg.norm(com[:2] - pelvis_pos[:2])
            return 1.0 / (1.0 + offset)
        except mujoco.MujocoError:
            return 0.0
    
    def get_body_velocity_norm(self, body_name):
        try:
            body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, body_name)
            vel = self.data.cvel[body_id][:3]  # Linear velocity only
            return np.linalg.norm(vel)
        except mujoco.MujocoError:
            return 0.0


    def ik_imitation_reward(self, smplx_frame):
        joint_weights = {
            "left_hip": 1.0, "right_hip": 1.0,
            "left_knee": 1.5, "right_knee": 1.5,
            "left_ankle": 1.0, "right_ankle": 1.0,
            "left_shoulder": 1.0, "right_shoulder": 1.0,
            "left_elbow": 1.5, "right_elbow": 1.5,
            "left_wrist": 2.0, "right_wrist": 2.0,
            "pelvis": 0.5,
            "spine1": 0.3, "spine2": 0.3, "spine3": 0.3,
            "neck": 0.3, "head": 0.3
        }

        total_loss = 0.0
        total_weight = 0.0
        for idx, smplx_name in enumerate(self.joint_names):
            if smplx_name not in self.joint_mapping:
                continue
            smplx_pos = smplx_frame[idx]
            weight = joint_weights.get(smplx_name, 1.0)
            mapped = self.joint_mapping[smplx_name]
            if isinstance(mapped, list):
                for mjoint in mapped:
                    mj_pos = self.get_body_pos(mjoint)
                    if mj_pos is not None:
                        total_loss += weight * np.linalg.norm(mj_pos - smplx_pos) ** 2
                        total_weight += weight
            else:
                mj_pos = self.get_body_pos(mapped)
                if mj_pos is not None:
                    total_loss += weight * np.linalg.norm(mj_pos - smplx_pos) ** 2
                    total_weight += weight

        imitation_loss = total_loss / total_weight if total_weight > 0 else 0.0
        upright_reward = 1.5 * self.compute_upright_reward()
        com_reward = self.compute_com_reward()

        # ✅ Add curiosity reward
        curiosity_beta = 0.01  # Small beta value; adjust based on your results
        curiosity_reward = 0.0
        for joint_name in self.joint_mapping.values():
            if isinstance(joint_name, list):
                for name in joint_name:
                    curiosity_reward += self.get_body_velocity_norm(name)
            else:
                curiosity_reward += self.get_body_velocity_norm(joint_name)
        curiosity_reward *= curiosity_beta

        # ✅ Final combined reward
        reward = (
            -self.imitation_weight * imitation_loss
            + 3.0 * upright_reward
            + 2.0 * com_reward
            + curiosity_reward
        )
        return reward

    def get_body_pos(self, body_name):
        try:
            body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, body_name)
            return self.data.xpos[body_id]
        except mujoco.MujocoError:
            return None

    def visualize_reference_and_pose(self):
        if self.frame_idx >= len(self.reference_motion):
            print("No frame to visualize.")
            return

        ref_frame = self.reference_motion[self.frame_idx]
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # Reference motion
        ref_xyz = np.array([ref_frame[i] for i in range(len(ref_frame))])
        ax1.scatter(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], c='b')
        ax1.set_title("Reference Motion")
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([0, 2])

        # MuJoCo pose
        pose_xyz = []
        for name in self.joint_mapping.values():
            if isinstance(name, list):
                for n in name:
                    pos = self.get_body_pos(n)
                    if pos is not None:
                        pose_xyz.append(pos)
            else:
                pos = self.get_body_pos(name)
                if pos is not None:
                    pose_xyz.append(pos)
        pose_xyz = np.array(pose_xyz)
        ax2.scatter(pose_xyz[:, 0], pose_xyz[:, 1], pose_xyz[:, 2], c='r')
        ax2.set_title("MuJoCo Pose")
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([0, 2])

        plt.tight_layout()
        plt.show()
