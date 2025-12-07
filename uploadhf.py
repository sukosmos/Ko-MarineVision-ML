from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="/data/CodeLLM/ML/src/vit/outputs/vit/checkpoints", repo_id="sukosmos/K-Marine-ViT", repo_type="model")
