from humanoid_imitation_env import HumanoidImitationEnv
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import joblib

# Load SMPLX pose sequence
res = joblib.load("walk.pt")
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
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,               # Small LR helps stabilize training on complex motions
    buffer_size=1_000_000,           # Large buffer to handle diverse motions
    learning_starts=25_000,          # Let the replay buffer fill up a bit before training
    batch_size=512,                  # Large batch size stabilizes gradient estimates
    tau=0.005,                       # Target smoothing coefficient (common for SAC/TQC)
    gamma=0.99,                      # High discount factor is important for long-horizon tasks
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    ent_coef="auto_0.1",             # Conservative entropy coefficient for imitation
    top_quantiles_to_drop_per_net=2,# From TQC paper (4->2 quantiles dropped improves performance)
    policy_kwargs=dict(
        net_arch=[512, 512, 512],    # Deep MLP works better for high-dim control
        log_std_init=-2,             # Conservative policy std init
        use_sde=True,                # State-dependent exploration helps in locomotion
    ),
    verbose=1
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
