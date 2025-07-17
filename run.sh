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
    echo "  1. Real AI Chat (Complete CLI)"
    echo "  2. Simple Demo Chat"
    echo "  3. Real AI Reasoning"
    echo "  4. Interactive AI Chat"
    echo "  5. System Info"
    echo "  6. View AI Models"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            read -p "Enter your question: " question
            echo -e "${GREEN}Using real AI...${NC}"
            python complete_cli.py chat "$question"
            ;;
        2)
            echo -e "${GREEN}Starting simple demo chat...${NC}"
            python simple_cli.py chat --interactive
            ;;
        3)
            read -p "Enter a problem to solve: " problem
            echo -e "${GREEN}Using advanced AI reasoning...${NC}"
            python complete_cli.py reasoning "$problem"
            ;;
        4)
            echo -e "${GREEN}Starting real AI interactive chat...${NC}"
            python complete_cli.py chat --interactive
            ;;
        5)
            python complete_cli.py info
            ;;
        6)
            python complete_cli.py models
            ;;
        *)
            echo "Invalid option. Starting real AI chat..."
            python complete_cli.py chat --interactive
            ;;
    esac
else
    # Arguments passed, determine which CLI to use
    if [ "$1" = "complete" ]; then
        shift  # Remove 'complete' from arguments
        echo -e "${GREEN}Running complete CLI: python complete_cli.py $@${NC}"
        python complete_cli.py "$@"
    else
        echo -e "${GREEN}Running simple CLI: python simple_cli.py $@${NC}"
        python simple_cli.py "$@"
    fi
fi