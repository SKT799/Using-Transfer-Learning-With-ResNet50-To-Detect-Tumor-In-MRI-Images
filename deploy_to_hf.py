import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# Use the token from environment
TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "DumbMaddy"
SPACE_NAME = "Brain-Tumor-Detection-MRI"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

api = HfApi(token=TOKEN)

print(f"Creating Space {REPO_ID}...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    print("Space created or already exists.")
except Exception as e:
    print(f"Error creating space: {e}")

print("Uploading files to Space...")
try:
    # Upload everything in the current directory except some unnecessary folders
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=["deploy_to_hf.py", ".env", ".git*", "__pycache__*", "uploads/", "dataset/", "*.ipynb"]
    )
    print("Files uploaded successfully!")
    print(f"Your app should be deploying at: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"Error uploading files: {e}")
