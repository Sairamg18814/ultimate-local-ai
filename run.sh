#!/bin/bash
# Ultimate Local AI - Quick Run Script

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}🤖 Ultimate Local AI - Quick Launcher${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Running setup...${NC}"
    ./setup.sh
    echo ""
fi

# Activate virtual environment
echo -e "${GREEN}✅ Activating virtual environment...${NC}"
source venv/bin/activate

# Check if arguments were passed
if [ $# -eq 0 ]; then
    # No arguments, show menu
    echo -e "${CYAN}🎯 Quick Commands:${NC}"
    echo "  1. Ultimate AI Chat (Best)"
    echo "  2. Interactive AI Chat"
    echo "  3. AI Reasoning Engine"
    echo "  4. Simple Demo"
    echo "  5. System Info"
    echo "  6. View AI Models"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            read -p "Enter your question: " question
            echo -e "${GREEN}Using Ultimate AI...${NC}"
            python ultimate_cli.py chat "$question"
            ;;
        2)
            echo -e "${GREEN}Starting interactive AI chat...${NC}"
            python ultimate_cli.py chat --interactive
            ;;
        3)
            read -p "Enter a problem to solve: " problem
            echo -e "${GREEN}Using AI reasoning engine...${NC}"
            python ultimate_cli.py reasoning "$problem"
            ;;
        4)
            echo -e "${GREEN}Starting simple demo...${NC}"
            python simple_cli.py chat --interactive
            ;;
        5)
            python ultimate_cli.py info
            ;;
        6)
            python ultimate_cli.py models
            ;;
        *)
            echo "Invalid option. Starting Ultimate AI chat..."
            python ultimate_cli.py chat --interactive
            ;;
    esac
else
    # Arguments passed, determine which CLI to use
    if [ "$1" = "ultimate" ]; then
        shift  # Remove 'ultimate' from arguments
        echo -e "${GREEN}Running Ultimate AI: python ultimate_cli.py $@${NC}"
        python ultimate_cli.py "$@"
    elif [ "$1" = "simple" ]; then
        shift  # Remove 'simple' from arguments
        echo -e "${GREEN}Running simple demo: python simple_cli.py $@${NC}"
        python simple_cli.py "$@"
    else
        # Default to Ultimate AI
        echo -e "${GREEN}Running Ultimate AI: python ultimate_cli.py $@${NC}"
        python ultimate_cli.py "$@"
    fi
fi