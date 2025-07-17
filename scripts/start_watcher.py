#!/usr/bin/env python3
"""
File Watcher Starter Script
Starts the file watcher for automatic GitHub updates
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🔄 Starting Ultimate Local AI File Watcher...")
    print("This will automatically commit and push changes to GitHub.")
    print("Press Ctrl+C to stop the watcher.\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    try:
        # Start the GitHub integration script in watch mode
        subprocess.run([sys.executable, "scripts/github_integration.py", "--watch"])
    except KeyboardInterrupt:
        print("\n👋 File watcher stopped.")
    except Exception as e:
        print(f"❌ Error starting file watcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()