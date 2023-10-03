import os


# Get the base directory of the current file
base_dir = os.path.dirname(os.path.dirname(__file__))

# Create the path to the 'data' directory
data_dir = os.path.join(base_dir, "data")

# Create the path to the 'checkpoint' directory
checkpoint_dir = os.path.join(base_dir, "weights", "checkpoint")

# Create the path to the 'vit_tiny_config.json' file
config_file = os.path.join(base_dir, "vit_tiny_config.json")
