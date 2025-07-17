#!/usr/bin/env python3
"""
GitHub Integration Script for Ultimate Local AI
Automatically creates GitHub repository and sets up continuous deployment
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/github_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubIntegration:
    """Handles GitHub repository creation and automatic updates"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_owner = os.getenv('GITHUB_OWNER', 'your-username')
        self.github_repo = os.getenv('GITHUB_REPO', 'ultimate-local-ai')
        self.github_branch = os.getenv('GITHUB_BRANCH', 'main')
        
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        self.base_url = 'https://api.github.com'
        
    def check_repo_exists(self) -> bool:
        """Check if the repository already exists"""
        url = f"{self.base_url}/repos/{self.github_owner}/{self.github_repo}"
        response = requests.get(url, headers=self.headers)
        return response.status_code == 200
    
    def create_repository(self, description: str = None, private: bool = False) -> bool:
        """Create a new GitHub repository"""
        if self.check_repo_exists():
            logger.info(f"Repository {self.github_owner}/{self.github_repo} already exists")
            return True
        
        url = f"{self.base_url}/user/repos"
        
        repo_data = {
            'name': self.github_repo,
            'description': description or 'Ultimate Local AI CLI - Advanced Local AI with Adaptive Intelligence',
            'private': private,
            'auto_init': False,
            'has_issues': True,
            'has_projects': True,
            'has_wiki': True
        }
        
        logger.info(f"Creating repository {self.github_repo}...")
        response = requests.post(url, headers=self.headers, json=repo_data)
        
        if response.status_code == 201:
            logger.info("Repository created successfully!")
            return True
        else:
            logger.error(f"Failed to create repository: {response.status_code} - {response.text}")
            return False
    
    def setup_git_remote(self) -> bool:
        """Setup git remote origin"""
        try:
            # Check if remote already exists
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Remote origin already configured")
                return True
            
            # Add remote origin
            remote_url = f"https://github.com/{self.github_owner}/{self.github_repo}.git"
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
            logger.info(f"Added remote origin: {remote_url}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup git remote: {e}")
            return False
    
    def create_initial_commit(self) -> bool:
        """Create initial commit with all files"""
        try:
            # Add all files
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Create initial commit
            commit_message = """Initial commit: Ultimate Local AI CLI

🚀 Features:
- Adaptive Intelligence Controller
- Real-Time RAG Pipeline
- Advanced Reasoning Engine
- 4-Tier Memory System
- Continuous Learning with LoRA
- MLX Optimization for Apple Silicon
- Rich CLI Interface

🤖 Generated with Ultimate Local AI"""
            
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Set default branch to main
            subprocess.run(['git', 'branch', '-M', 'main'], check=True)
            
            logger.info("Initial commit created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create initial commit: {e}")
            return False
    
    def push_to_github(self) -> bool:
        """Push the repository to GitHub"""
        try:
            # Configure git credentials for this push
            remote_url = f"https://{self.github_token}@github.com/{self.github_owner}/{self.github_repo}.git"
            subprocess.run(['git', 'remote', 'set-url', 'origin', remote_url], check=True)
            
            # Push to GitHub
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
            
            logger.info("Successfully pushed to GitHub!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to GitHub: {e}")
            return False
    
    def setup_github_actions(self) -> bool:
        """Setup GitHub Actions for CI/CD"""
        try:
            # Create .github/workflows directory
            workflows_dir = Path('.github/workflows')
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CI workflow
            ci_workflow = """name: Ultimate Local AI CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        ruff check .
        black --check .
    
    - name: Run type checking
      run: |
        mypy ultimate_local_ai/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=ultimate_local_ai/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: github/super-linter@v4
      env:
        DEFAULT_BRANCH: main
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        VALIDATE_PYTHON: true
        VALIDATE_PYTHON_BLACK: true
        VALIDATE_PYTHON_ISORT: true
"""
            
            with open(workflows_dir / 'ci.yml', 'w') as f:
                f.write(ci_workflow)
            
            logger.info("GitHub Actions workflow created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup GitHub Actions: {e}")
            return False

class FileWatcher(FileSystemEventHandler):
    """Watches for file changes and auto-commits"""
    
    def __init__(self, github_integration: GitHubIntegration):
        self.github_integration = github_integration
        self.last_commit_time = time.time()
        self.commit_interval = int(os.getenv('COMMIT_INTERVAL', '300'))  # 5 minutes default
        self.auto_commit = os.getenv('AUTO_COMMIT', 'true').lower() == 'true'
        self.auto_push = os.getenv('AUTO_PUSH', 'true').lower() == 'true'
        self.watch_paths = os.getenv('WATCH_PATHS', 'ultimate_local_ai/,*.py,*.md,*.yaml,*.json').split(',')
        
    def should_watch_file(self, file_path: str) -> bool:
        """Check if file should be watched for changes"""
        file_path = Path(file_path)
        
        # Skip certain directories and files
        skip_patterns = ['.git', '__pycache__', '.pytest_cache', 'logs', 'cache', 'models', '.env']
        if any(pattern in str(file_path) for pattern in skip_patterns):
            return False
        
        # Check watch patterns
        for pattern in self.watch_paths:
            pattern = pattern.strip()
            if pattern.endswith('/'):
                # Directory pattern
                if pattern[:-1] in str(file_path):
                    return True
            elif '*' in pattern:
                # Glob pattern
                if file_path.match(pattern):
                    return True
            elif str(file_path).endswith(pattern):
                return True
        
        return False
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if not self.should_watch_file(event.src_path):
            return
        
        current_time = time.time()
        if current_time - self.last_commit_time >= self.commit_interval:
            self.auto_commit_changes()
            self.last_commit_time = current_time
    
    def auto_commit_changes(self):
        """Automatically commit and push changes"""
        if not self.auto_commit:
            return
        
        try:
            # Check if there are changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            
            if not result.stdout.strip():
                return  # No changes
            
            # Add changes
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Create commit
            commit_message = f"""Auto-update: {time.strftime('%Y-%m-%d %H:%M:%S')}

Automatic commit of latest changes

🤖 Generated with Ultimate Local AI"""
            
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            if self.auto_push:
                subprocess.run(['git', 'push'], check=True)
                logger.info("Auto-committed and pushed changes")
            else:
                logger.info("Auto-committed changes (push disabled)")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to auto-commit: {e}")

def main():
    """Main function to setup GitHub integration"""
    try:
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize GitHub integration
        github = GitHubIntegration()
        
        print("🚀 Setting up Ultimate Local AI GitHub Integration...")
        
        # Step 1: Create GitHub repository
        if not github.create_repository():
            sys.exit(1)
        
        # Step 2: Setup git remote
        if not github.setup_git_remote():
            sys.exit(1)
        
        # Step 3: Create initial commit
        if not github.create_initial_commit():
            sys.exit(1)
        
        # Step 4: Setup GitHub Actions
        if not github.setup_github_actions():
            logger.warning("GitHub Actions setup failed, continuing anyway...")
        
        # Step 5: Push to GitHub
        if not github.push_to_github():
            sys.exit(1)
        
        print(f"✅ Successfully created and pushed to GitHub!")
        print(f"📝 Repository: https://github.com/{github.github_owner}/{github.github_repo}")
        
        # Step 6: Setup file watching (optional)
        if len(sys.argv) > 1 and sys.argv[1] == '--watch':
            print("👀 Starting file watcher for auto-commits...")
            
            event_handler = FileWatcher(github)
            observer = Observer()
            observer.schedule(event_handler, '.', recursive=True)
            observer.start()
            
            try:
                print("File watcher started. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
                print("\nFile watcher stopped.")
            
            observer.join()
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()