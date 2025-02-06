import torch
import numpy as np
import os
from VAE import VAE
from Data_Collection.gym_data_collection import load_data
import config

# get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))


def compute_dataset_statistics(data_path):
    """
    Compute global mean and standard deviation across the entire dataset
    """
    # Load dataset
    episodes, _, _ = load_data(path=data_path)

    # Concatenate all data into a single tensor
    all_data = torch.tensor([state for episode in episodes for state in episode], dtype=torch.float32)

    # Compute global statistics (single values)
    mean = all_data.mean()
    std = torch.max(all_data.std(), torch.tensor(1e-8))

    return mean.item(), std.item()

def compute_standardized_mse(predicted, target, global_mean, global_std):
    """
    Compute standardized MSE between predicted and target tensors
    """
    # Standardize both tensors
    target_mean = target.mean(dim=1, keepdim=True)
    target_std = target.std(dim=1, keepdim=True) + 1e-8

    standardized_pred = (predicted - global_mean) / global_std
    standardized_target = (target - global_mean) / global_std

    # Compute MSE
    mse = torch.mean((standardized_pred - standardized_target) ** 2)
    return mse.item()


def evaluate_vae(vae_path, data_path, eval_frequency=5, output_path="vae_scores.txt"):
    """
    Evaluate VAE performance on a single dataset

    Args:
        vae_path: Path to trained VAE model
        data_path: Path to dataset file
        eval_frequency: Frequency of evaluation in sliding window
        output_path: Path to save results
    """
    # Load VAE
    vae = VAE(input_dim=config.INPUT_DIMENSION,
              latent_dim=config.LATENT_DIM,
              output_dim=config.OUTPUT_DIMENSION)
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()

    # Load dataset
    episodes, _, _ = load_data(path=data_path)
    total_mse = 0
    total_samples = 0

    #compute global mean and std for data set :
    global_mean, global_std = compute_dataset_statistics(data_path=data_path)

    for episode in episodes:
        # Sliding window approach
        for idx in range(0, len(episode) - config.INPUT_STATE_SIZE - config.OUTPUT_STATE_SIZE, eval_frequency):
            # Get input states
            input_states = np.concatenate(
                list(episode)[idx:idx + config.INPUT_STATE_SIZE],
                axis=-1
            )

            # Get target output states
            target_states = np.concatenate(
                list(episode)[idx + config.INPUT_STATE_SIZE:
                              idx + config.INPUT_STATE_SIZE + config.OUTPUT_STATE_SIZE],
                axis=-1
            )

            # Convert to tensors
            input_tensor = torch.tensor(input_states, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor(target_states, dtype=torch.float32).unsqueeze(0)

            # Get prediction
            with torch.no_grad():
                predicted_states, _, _, _ = vae(input_tensor)

            # Compute standardized MSE
            mse = compute_standardized_mse(predicted_states, target_tensor, global_mean=global_mean, global_std=global_std)
            total_mse += mse
            total_samples += 1

    # Average MSE for dataset
    avg_mse = total_mse / total_samples

    # Save results
    with open(output_path, 'w') as f:
        f.write(f"VAE Model: {os.path.basename(vae_path)}\n")
        f.write(f"Dataset Path: {data_path}\n")
        f.write(f"Evaluation Frequency: {eval_frequency}\n\n")
        f.write("Results:\n" + "-" * 50 + "\n")
        f.write(f"Average Standardized MSE: {avg_mse:.6f}\n")
        f.write(f"Total Standardized MSE: {total_mse:.6f}\n")

    return avg_mse, total_mse


def vae_score_call(data_path=os.path.join(base_dir, "..", "Data_Collection", "collected_data", "1000_rand_Eval","random_1000_20250130_122312.npz"), vae_name='vae_rand_100k'):
    # Example usage

    #data_dir = os.path.join(base_dir, "..", "Data_Collection", "collected_data")


    # Path to VAE model
    vae_path = os.path.join(base_dir, "..", "VAE_pretrain", "pretrained_vae", config.VAE_Version,
                              f"{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}", f"KL-D_{config.BETA_KL_DIV}", vae_name)
    # define output path :
    output_dir = os.path.join(base_dir,"VAE_score", config.VAE_Version , f"{config.INPUT_STATE_SIZE}_{config.OUTPUT_STATE_SIZE}",
                            f"KL-D_{config.BETA_KL_DIV}")
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate
    avg_mse , total_mse = evaluate_vae(vae_path=vae_path, data_path=data_path, eval_frequency=5, output_path=os.path.join(output_dir, f"{vae_name}.txt"))
    print("Evaluation completed. Results saved to vae_scores.txt \n")
    print(f"avg_mse = {avg_mse} \n")
    print(f"total_mse = {total_mse} \n")



if __name__ == "__main__":
    vae_score_call()