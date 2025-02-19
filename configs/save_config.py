import os
from configs import config, eval_config


# get base_dir path
base_dir = os.path.dirname(os.path.abspath(__file__))
def save_config(config_name):

    # Extract all variables dynamically (excluding built-in attributes)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
    # Define the directory where the config file will be saved
    save_directory =  os.path.join(base_dir, "..","VAE_pretrain", "pretrained_vae", f'{config_dict["VAE_Version"]}' , f'{config_dict["INPUT_STATE_SIZE"]}_{config_dict["OUTPUT_STATE_SIZE"]}', f'KL-D_{config_dict["BETA_KL_DIV"]}' )  # <-- Change this to your desired path
    os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

    # Convert to a formatted string
    config_data = "\n".join([f"{key} = {value}" for key, value in config_dict.items()])

    # Define the file path
    file_path = os.path.join(save_directory, f"VAE_config_{config_name}.txt")

    # Write the config data to the file
    with open(file_path, "w") as file:
        file.write(config_data)

    print(f"Configuration saved to: {file_path}")


def save_vae_code():
    # Path to VAE.py
    vae_file_path = os.path.join(base_dir,"..", "VAE.py")

    # Extract all variables dynamically (excluding built-in attributes)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
    # Define the directory where the config file will be saved
    save_directory =  os.path.join(base_dir, "..","VAE_pretrain", "pretrained_vae", f'{config_dict["VAE_Version"]}' )  # <-- Change this to your desired path
    os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

    # Define the file path
    vae_save_path = os.path.join(save_directory, "VAE_code.txt")

    try:
        # Read and write the VAE.py content to the new file
        with open(vae_file_path, "r") as vae_file:
            vae_code = vae_file.read()

        with open(vae_save_path, "w") as file:
            file.write(vae_code)

        print(f"VAE code saved to: {vae_save_path}")
    except FileNotFoundError:
        print(f"Error: {vae_file_path} not found.")

def save_eval_config(batch_folder):

    # Extract all variables dynamically (excluding built-in attributes)
    config_dict = {k: v for k, v in vars(eval_config).items() if not k.startswith("__")}
    # Define the directory where the config file will be saved
    save_directory =  batch_folder
    os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

    # Convert to a formatted string
    config_data = "\n".join([f"{key} = {value}" for key, value in config_dict.items()])

    # Define the file path
    file_path = os.path.join(save_directory, "eval_config.txt")

    # Write the config data to the file
    with open(file_path, "w") as file:
        file.write(config_data)

    print(f"Configuration saved to: {file_path}")


if __name__ == "__main__" :
    save_config()
