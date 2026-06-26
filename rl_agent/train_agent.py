import sys
from pathlib import Path

# Ensure project root is on the path so `rl_agent.environment` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rl_agent.environment  # triggers gym.register()
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

SAVE_DIR = PROJECT_ROOT / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Create environment
env = gym.make("NetworkEnv-v0")
print(f"Environment: {env}")
print(f"  Observation space : {env.observation_space}")
print(f"  Action space      : {env.action_space}")

# DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    seed=42,
)

print("\nTraining DQN agent for 50,000 timesteps...")
model.learn(total_timesteps=50_000)

# Evaluate
eval_env = gym.make("NetworkEnv-v0")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"\nEvaluation over 10 episodes:")
print(f"  Mean reward : {mean_reward:.2f}")
print(f"  Std reward  : {std_reward:.2f}")

# Save
model_path = SAVE_DIR / "dqn_agent"
model.save(model_path)
print(f"\nModel saved to: {model_path}.zip")
print("\nRL Agent Training Complete")
