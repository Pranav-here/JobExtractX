#!/usr/bin/env python
import os
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, whoami, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.errors import HfHubHTTPError

def verify_token_permissions(token=None):
    """Verify that the token has the correct permissions."""
    try:
        user_info = whoami(token=token)
        print(f"Successfully authenticated as: {user_info['name']} ({user_info.get('fullname', '')})")
        
        # Get organizations - compatible with older huggingface_hub versions
        org_names = []
        try:
            # Try newer API method
            api = HfApi(token=token)
            if hasattr(api, 'list_organizations'):
                orgs = api.list_organizations()
                org_names = [org.name for org in orgs]
            else:
                print("Note: Unable to list organizations with this version of huggingface_hub")
                print("You can still upload to repositories under your username")
        except Exception as e:
            print(f"Note: Unable to list organizations: {e}")
            print("You can still upload to repositories under your username")
        
        if org_names:
            print(f"You can publish to these organizations: {', '.join(org_names)}")
        
        return user_info['name'], org_names
    except Exception as e:
        print(f"Error verifying token: {e}")
        print("Please make sure your token has write access.")
        return None, []

def upload_mistral_model_to_hub(model_path, hub_model_id, base_model="mistralai/Mistral-7B-Instruct-v0.3", 
                               token=None, commit_message=None, create_repo_first=False, private=False):
    """
    Upload a trained Mistral model with LoRA adapters to the Hugging Face Hub.
    
    Args:
        model_path (str): Path to the saved LoRA adapter weights
        hub_model_id (str): Hugging Face Hub model ID (e.g. 'username/model-name')
        base_model (str): Base model identifier to load first before applying LoRA weights
        token (str, optional): Hugging Face API token
        commit_message (str, optional): Commit message for the upload
        create_repo_first (bool): Whether to try creating the repository before pushing
        private (bool): Whether to create a private repository
    """
    print(f"Loading base model {base_model} and applying LoRA adapters from {model_path}")
    
    # Load model and tokenizer
    try:
        # First load the base model
        print(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Then apply the LoRA adapter weights
        print(f"Applying LoRA adapter weights from: {model_path}")
        model = PeftModel.from_pretrained(
            base_model_obj,
            model_path
        )
        
        # Optionally merge weights for efficiency if desired
        # Uncomment the line below to merge LoRA weights with base model
        # model = model.merge_and_unload()
        
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)
    
    # Login to Hugging Face Hub
    if token:
        login(token=token)
    else:
        # Try to get token from environment variable
        env_token = os.getenv("HF_TOKEN")
        if env_token:
            login(token=env_token)
            token = env_token
        else:
            # Try using the cached token from huggingface-cli login
            login()
    
    # Verify token permissions
    username, orgs = verify_token_permissions(token)
    if not username:
        print("Authentication failed. Please check your token.")
        sys.exit(1)
    
    # Default to username namespace if not specified
    if '/' not in hub_model_id:
        print(f"No namespace specified, using your username: {username}")
        hub_model_id = f"{username}/{hub_model_id}"
        print(f"Repository will be: {hub_model_id}")
    
    # Verify the repository namespace
    namespace = hub_model_id.split('/')[0]
    if namespace != username and namespace not in orgs:
        print(f"Warning: You are trying to push to namespace '{namespace}' but you may not have permissions for it.")
        print(f"You can definitely push to: '{username}'")
        correct = input(f"Would you like to use '{username}/{hub_model_id.split('/')[1]}' instead? (y/n): ")
        if correct.lower() == 'y':
            hub_model_id = f"{username}/{hub_model_id.split('/')[1]}"
        else:
            print("Continuing with original namespace. If you get a permission error, try again with your username.")
    
    # Default commit message
    if commit_message is None:
        commit_message = f"Upload Mistral model with LoRA fine-tuning"
    
    # Create repository first if requested
    if create_repo_first:
        try:
            print(f"Creating repository: {hub_model_id}")
            api = HfApi(token=token)
            try:
                api.create_repo(
                    repo_id=hub_model_id,
                    private=private,
                    exist_ok=True
                )
            except TypeError:
                # For older versions
                print("Note: Using older huggingface_hub API for repository creation")
                try:
                    api.create_repo(repo_id=hub_model_id, private=private)
                except Exception:
                    print("Note: Repository might already exist, continuing with upload")
            print(f"Repository created or already exists: {hub_model_id}")
        except Exception as e:
            print(f"Error creating repository: {e}")
            print("Attempting to push anyway in case the repository already exists...")
    
    print(f"Uploading model to Hugging Face Hub: {hub_model_id}")
    
    # Push model and tokenizer to Hub
    try:
        model.push_to_hub(hub_model_id, token=token, commit_message=commit_message, private=private)
        tokenizer.push_to_hub(hub_model_id, token=token, commit_message=commit_message, private=private)
        
        print(f"Model and tokenizer successfully uploaded to: https://huggingface.co/{hub_model_id}")
        print("You can now use this model directly with the Transformers library.")
    except HfHubHTTPError as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        if "401" in str(e):
            print("Authentication error. Please check your token has write access.")
        elif "403" in str(e):
            print(f"Permission denied. You don't have permission to push to {hub_model_id}.")
            print(f"You can only push to repositories under your username ({username}) or organizations you belong to.")
        elif "404" in str(e):
            print(f"Repository not found. Try creating it first with --create_repo")
        else:
            print("For help, visit: https://huggingface.co/docs/hub/security-tokens")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("If you're getting permission errors, try using --create_repo flag")

def main():
    parser = argparse.ArgumentParser(description="Upload a Mistral model with LoRA adapters to Hugging Face Hub")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the saved LoRA adapter weights"
    )
    
    parser.add_argument(
        "--hub_model_id", 
        type=str,
        required=True,
        help="Hugging Face Hub model ID (e.g. 'username/model-name')"
    )
    
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)"
    )
    
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face API token (optional if already logged in)"
    )
    
    parser.add_argument(
        "--commit_message", 
        type=str, 
        default=None,
        help="Commit message for the upload"
    )
    
    parser.add_argument(
        "--create_repo", 
        action="store_true",
        help="Create the repository before pushing if it doesn't exist"
    )
    
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Create a private repository"
    )
    
    parser.add_argument(
        "--verify_only", 
        action="store_true",
        help="Only verify the token permissions without uploading"
    )
    
    parser.add_argument(
        "--merge_weights",
        action="store_true",
        help="Merge LoRA weights into the base model before uploading"
    )
    
    args = parser.parse_args()
    
    # If only verifying, just check the token permissions and exit
    if args.verify_only:
        username, orgs = verify_token_permissions(args.token)
        if username:
            print(f"Token is valid for user: {username}")
            print(f"You can create repositories as: {username}")
            if orgs:
                print(f"You can also create repositories under these organizations: {', '.join(orgs)}")
        sys.exit(0)
    
    upload_mistral_model_to_hub(
        model_path=args.model_path,
        hub_model_id=args.hub_model_id,
        base_model=args.base_model,
        token=args.token,
        commit_message=args.commit_message,
        create_repo_first=args.create_repo,
        private=args.private
    )

if __name__ == "__main__":
    main()