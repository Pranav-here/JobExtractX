#!/usr/bin/env python
import os
import argparse
import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
            # Check if the list_organizations method exists
            if hasattr(api, 'list_organizations'):
                orgs = api.list_organizations()
                org_names = [org.name for org in orgs]
            else:
                # For older versions, we can't easily get organizations
                # But we'll still be able to use the username
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

def upload_model_to_hub(model_path, hub_model_id, token=None, commit_message=None, create_repo_first=False, private=False):
    """
    Upload a trained T5 model to the Hugging Face Hub.
    
    Args:
        model_path (str): Path to the saved model
        hub_model_id (str): Hugging Face Hub model ID (e.g. 'username/model-name')
        token (str, optional): Hugging Face API token. If None, will use the token from
                              huggingface-cli login or HF_TOKEN env variable.
        commit_message (str, optional): Commit message for the upload.
        create_repo_first (bool): Whether to try creating the repository before pushing
        private (bool): Whether to create a private repository
    """
    print(f"Loading model and tokenizer from {model_path}")
    
    # Load model and tokenizer
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
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
        commit_message = "Upload trained T5 model for long sequences"
    
    # Create repository first if requested
    if create_repo_first:
        try:
            print(f"Creating repository: {hub_model_id}")
            api = HfApi(token=token)
            # Use the create_repo method with appropriate arguments based on availability
            try:
                api.create_repo(
                    repo_id=hub_model_id,
                    private=private,
                    exist_ok=True
                )
            except TypeError:
                # Older versions might not have exist_ok or other parameters
                print("Note: Using older huggingface_hub API for repository creation")
                try:
                    api.create_repo(repo_id=hub_model_id, private=private)
                except Exception:
                    # Repository might already exist
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
    parser = argparse.ArgumentParser(description="Upload a trained T5 model to Hugging Face Hub")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./t5_model_final",
        help="Path to the saved model (default: ./t5_model_final)"
    )
    
    parser.add_argument(
        "--hub_model_id", 
        type=str,
        help="Hugging Face Hub model ID (e.g. 'username/model-name')"
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
    
    # Otherwise, the hub_model_id is required
    if not args.hub_model_id:
        parser.error("the following arguments are required: --hub_model_id")
    
    upload_model_to_hub(
        model_path=args.model_path,
        hub_model_id=args.hub_model_id,
        token=args.token,
        commit_message=args.commit_message,
        create_repo_first=args.create_repo,
        private=args.private
    )

if __name__ == "__main__":
    main() 