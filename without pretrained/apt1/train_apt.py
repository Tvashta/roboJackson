from humanoid_imitation_env import HumanoidImitationEnv
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import joblib

# Load SMPLX pose sequence
res = joblib.load("apt1.pt")
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

# Initialize TQC model from scratch (no pretrained weights)
model = TQC(
    MlpPolicy,
    env,
    verbose=1,
    learning_rate=3e-4,       # You can tune this
    buffer_size=1000000,
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_entropy="auto",
)

# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="tqc_imitation_fromscratch"
)

# Train from scratch
model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback
)

# Save the final model
model.save("apt_tqc_imitation_fromscratch_final")
