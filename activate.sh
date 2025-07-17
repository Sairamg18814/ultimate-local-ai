#!/bin/bash
# Ultimate Local AI - Virtual Environment Activation Script

echo "🤖 Activating Ultimate Local AI virtual environment..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if activation was successful
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment activated!"
    echo "📍 Virtual environment: $VIRTUAL_ENV"
    echo ""
    echo "🎯 Quick Commands:"
    echo "  • python simple_cli.py chat --interactive"
    echo "  • python simple_cli.py reasoning \"your problem\""
    echo "  • python simple_cli.py info"
    echo "  • python simple_cli.py models"
    echo ""
    echo "🔄 GitHub Integration:"
    echo "  • python scripts/quick_update.py \"commit message\""
    echo "  • python scripts/start_watcher.py"
    echo ""
    echo "Type 'deactivate' to exit the virtual environment."
    
    # Start a new shell with the virtual environment activated
    exec $SHELL
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi