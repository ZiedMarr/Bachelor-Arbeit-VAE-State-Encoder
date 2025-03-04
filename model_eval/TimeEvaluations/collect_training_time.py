import os
import json
import pandas as pd


def collect_training_times(log_batch_dir, output_dir = "./"):
    """Collect all training times from process directories and create a summary."""
    all_times = []
    import os



    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all process directories
    for item in os.listdir(log_batch_dir):
        process_dir = os.path.join(log_batch_dir, item)
        if os.path.isdir(process_dir) and item.startswith("process_"):
            timing_file = os.path.join(process_dir, "training_time.json")
            if os.path.exists(timing_file):
                with open(timing_file, 'r') as f:
                    timing_data = json.load(f)
                    all_times.append(timing_data)

    # Convert to DataFrame for easy analysis
    if all_times:
        df = pd.DataFrame(all_times)

        # Calculate summary statistics
        avg_time = df['training_time_seconds'].mean()
        min_time = df['training_time_seconds'].min()
        max_time = df['training_time_seconds'].max()
        std_time = df['training_time_seconds'].std()

        # Save summary
        summary = {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_time,
            'total_processes': len(df)
        }

        # Save to CSV and summary JSON
        df.to_csv(os.path.join(output_dir, "all_training_times.csv"), index=False)
        with open(os.path.join(output_dir, "time_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        return df, summary

    return None, None

if __name__ == "__main__" :
    collect_training_times("../../VAE_PPO_train/logs/batch_1M_VAE_Version_6.23_vae_mix_10ep_config_v6_hexa_input_medium_latent_GB_4", output_dir= "./vae_6_2_VAE_PPO_train/logs/batch_1M_VAE_Version_6.23_vae_mix_10ep_config_v6_hexa_input_medium_latent_GB_4")