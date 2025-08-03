import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login, create_repo
from pathlib import Path

def upload_model_to_hub(
    model_path: str,
    repo_name: str,
    token: str = None,
    organization: str = None,
    private: bool = False,
    commit_message: str = "Upload fine-tuned model",
    model_card_template: str = None,
):
    """
    Upload a locally saved model to Hugging Face Hub.
    
    Args:
        model_path (str): Path to the local model directory
        repo_name (str): Name of the repository on Hugging Face Hub
        token (str, optional): Hugging Face authentication token. If not provided, will use the token from `huggingface-cli login`
        organization (str, optional): Organization name if uploading to an organization
        private (bool, optional): Whether the repository should be private
        commit_message (str, optional): Commit message for the upload
        model_card_template (str, optional): Path to a model card template (README.md)
    """
    # Verify model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Login to Hugging Face
    if token:
        login(token=token)
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository namespace
    repo_namespace = f"{organization}/{repo_name}" if organization else repo_name
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_namespace,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"Created repository: {repo_namespace}")
    except Exception as e:
        print(f"Using existing repository: {repo_namespace}")
    
    # Upload model files
    print(f"Uploading model from {model_path}...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_namespace,
        repo_type="model",
        commit_message=commit_message,
    )
    
    # Upload model card if provided
    if model_card_template and os.path.exists(model_card_template):
        with open(model_card_template, 'r', encoding='utf-8') as f:
            model_card = f.read()
        
        # Create or update README.md in the repository
        api.upload_file(
            path_or_fileobj=model_card.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_namespace,
            repo_type="model",
            commit_message="Add model card"
        )
    
    print(f"Successfully uploaded model to: https://huggingface.co/{repo_namespace}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the local model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Name of the repository on Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face authentication token")
    parser.add_argument("--organization", type=str, default=None,
                       help="Organization name (optional)")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--model_card", type=str, default=None,
                       help="Path to model card template (README.md)")
    
    args = parser.parse_args()
    
    upload_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        organization=args.organization,
        private=args.private,
        model_card_template=args.model_card
    )
