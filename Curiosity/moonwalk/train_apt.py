from humanoid_imitation_env import HumanoidImitationEnv
from sb3_contrib import TQC
import joblib
from stable_baselines3.common.callbacks import CheckpointCallback

# Load SMPLX pose sequence
res = joblib.load("moonwalk.pt")
reference_motion = res["pred_xyz_29"]
smplx_joint_names = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_eye", "right_eye", "left_big_toe",
    "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel", "nose"
]
reference_motion = reference_motion[:100]  # Optional limit

# Load imitation environment
env = HumanoidImitationEnv(reference_motion, smplx_joint_names)

# Load pretrained TQC walking model
model = TQC.load("humanoid-v5-TQC-simple.zip", env=env, verbose=1)

# (Optional) Adjust learning rate for fine-tuning
model.learning_rate = 0.01  # Lower to prevent overwriting pretrained knowledge

# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="tqc_imitation_norm"
)

# Fine-tune the model
model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback
)

# Save final model
model.save("apt_tqc_imitation_final2")
