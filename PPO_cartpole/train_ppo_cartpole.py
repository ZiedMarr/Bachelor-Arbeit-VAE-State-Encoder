import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from datetime import datetime
from Wrappers.RandomStartCartPole import RandomStartCartPole
import os

def train_ppo_cartpole(log_batch_dir, total_timesteps=20000, seed=42) :

    # Define the base directory (directory of the current script)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    #get timestamp :
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a log directory name
    log_dir = os.path.join(log_batch_dir,f"logs_{total_timesteps}_{timestamp}")


    # Create the evaluation environment
    eval_env = gym.make("CartPole-v1")
    eval_env.reset(seed=seed)
    eval_env = RandomStartCartPole(eval_env)
    eval_env = Monitor(eval_env)  # Monitor to log evaluation statistics

    # Initialize the PPO model
    model = PPO("MlpPolicy", eval_env, verbose=1)

    # Set up TensorBoard logger
    model.set_logger(configure(os.path.join(log_dir, "tensorboard_logs"), ["tensorboard"]))

    # Initialize checkpoint callback for saving models every 500 timesteps
    #checkpoint_callback = CheckpointCallback(save_freq=1, save_path="PPO_cartpole_trained/")
    #event_callback = EveryNTimesteps(n_steps=2048, callback=checkpoint_callback)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join( log_dir, "best_model"),
        log_path=os.path.join( log_dir, "eval"),
        eval_freq=2048,  # Evaluate every 10,000 timesteps
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback])



    eval_env.close()


    print("Training completed and model saved.")