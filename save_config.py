import os
import config  # Import your Python config file

# Define the directory where the config file will be saved
save_directory = "./VAE_pretrain/pretrained_vae/VAE_Version_"  # <-- Change this to your desired path
os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

# Extract all variables dynamically (excluding built-in attributes)
config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
# Define the directory where the config file will be saved
save_directory = f"./VAE_pretrain/pretrained_vae/{config_dict["VAE_Version"]}"  # <-- Change this to your desired path
os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

# Convert to a formatted string
config_data = "\n".join([f"{key} = {value}" for key, value in config_dict.items()])

# Define the file path
file_path = os.path.join(save_directory, "VAE_config.txt")

# Write the config data to the file
with open(file_path, "w") as file:
    file.write(config_data)

print(f"Configuration saved to: {file_path}")
