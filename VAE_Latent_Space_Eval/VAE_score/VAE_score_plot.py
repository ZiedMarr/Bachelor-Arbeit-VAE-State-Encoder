import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Base directory path
base_dir = "./"


# Data structure to store MSE values
mse_data = {

    "random": [],
    "exp_no_noise": [],
    "exp_0.3_noise": []
}


# Function to extract MSE from file content
def extract_mse(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r'Average Standardized MSE:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))
    return None


# Traverse the directory tree
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)


            # Determine file category
            if file.startswith('vae_exp_no_noise'):
                category = "exp_no_noise"
            elif file.startswith('vae_exp_0.3noise'):
                category = "exp_0.3_noise"
            elif file.startswith('vae_random_50k'):
                category = "random"
            else:
                continue

            # Extract MSE value
            mse = extract_mse(file_path)
            if mse is not None:
                mse_data[category].append(mse)

# Calculate average MSE for each category
avg_mse = {
    category: np.mean(values) if values else 0
    for category, values in mse_data.items()
}

# Create bar chart
categories = list(avg_mse.keys())
values = [avg_mse[cat] for cat in categories]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['blue', 'green', 'orange'])

# Add labels and title
plt.xlabel('Training data Type')
plt.ylabel('Average Standardized MSE')
plt.title('Average Standardized MSE by training data type')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig('vae_mse_comparison.png')
plt.show()