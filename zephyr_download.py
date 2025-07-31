import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
# Specify the Hugging Face model identifier you want to download
model_id = "stabilityai/stablelm-zephyr-3b"
# Specify the local directory where you want to save the model files
# (This path is relative to where you run this script)
# Example: './zephyr_local_model' will create a folder named 'zephyr_local_model'
# in the same directory as this script.
local_model_dir = "./zephyr_local_model"
# --- End Configuration ---

def download_model_locally(model_name_or_path, save_directory):
    """
    Downloads and saves a Hugging Face model and tokenizer to a local directory.
    """
    print(f"--- Starting download for model: {model_name_or_path} ---")
    print(f"Target local directory: {save_directory}")

    # Create the target directory if it doesn't exist
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")
        else:
            print(f"Directory already exists: {save_directory}")
    except OSError as e:
        print(f"❌ Error creating directory {save_directory}: {e}")
        return

    # Download and save the tokenizer
    try:
        print(f"Downloading tokenizer for {model_name_or_path}...")
        # Use trust_remote_code=True if required by the model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.save_pretrained(save_directory)
        print(f"✅ Tokenizer saved successfully to {save_directory}")
    except Exception as e:
        print(f"❌ Error downloading/saving tokenizer: {e}")
        print(traceback.format_exc())
        return # Stop if tokenizer fails

    # Download and save the model
    try:
        print(f"Downloading model {model_name_or_path} (this may take time and disk space)...")
        # Use trust_remote_code=True if required by the model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        model.save_pretrained(save_directory)
        print(f"✅ Model saved successfully to {save_directory}")
    except Exception as e:
        print(f"❌ Error downloading/saving model: {e}")
        print(traceback.format_exc())
        return

    print(f"--- Download complete for {model_name_or_path} ---")

# --- Main execution ---
if __name__ == "__main__":
    download_model_locally(model_id, local_model_dir)
    print("\n--- Next Steps ---")
    print(f"1. Ensure the folder '{local_model_dir}' contains the model files.")
    print(f"2. In your main bot script (e.g., DXEXTRACT_ZEPHYR3B.py), set:")
    print(f"   MODEL_PATH = '{local_model_dir}'")
