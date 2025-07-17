#!/usr/bin/env python3
"""
Quick Update Script
Manually commit and push changes to GitHub
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def main():
    """Quick commit and push changes"""
    try:
        # Change to project directory
        project_dir = Path(__file__).parent.parent
        os.chdir(project_dir)
        
        print("🚀 Quick Update: Committing and pushing changes...")
        
        # Check if there are changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if not result.stdout.strip():
            print("✅ No changes to commit.")
            return
        
        print(f"📝 Found changes:\n{result.stdout}")
        
        # Add changes
        subprocess.run(['git', 'add', '.'], check=True)
        print("✅ Added changes to staging area")
        
        # Get commit message from command line or use default
        if len(sys.argv) > 1:
            commit_message = ' '.join(sys.argv[1:])
        else:
            commit_message = f"Quick update: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create commit
        full_commit_message = f"""{commit_message}

🤖 Generated with Ultimate Local AI"""
        
        subprocess.run(['git', 'commit', '-m', full_commit_message], check=True)
        print("✅ Created commit")
        
        # Push to GitHub
        subprocess.run(['git', 'push'], check=True)
        print("✅ Pushed to GitHub")
        
        print(f"\n🎉 Successfully updated GitHub repository!")
        print(f"📝 Repository: https://github.com/Sairamg18814/ultimate-local-ai")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during update: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()