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
    echo "  1. Interactive Chat"
    echo "  2. Ask a Question"
    echo "  3. Reasoning Mode"
    echo "  4. System Info"
    echo "  5. Memory Stats"
    echo "  6. View Models"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}Starting interactive chat...${NC}"
            python simple_cli.py chat --interactive
            ;;
        2)
            read -p "Enter your question: " question
            python simple_cli.py chat "$question"
            ;;
        3)
            read -p "Enter a problem to solve: " problem
            python simple_cli.py reasoning "$problem"
            ;;
        4)
            python simple_cli.py info
            ;;
        5)
            python simple_cli.py memory --action stats
            ;;
        6)
            python simple_cli.py models
            ;;
        *)
            echo "Invalid option. Starting interactive chat..."
            python simple_cli.py chat --interactive
            ;;
    esac
else
    # Arguments passed, run directly
    echo -e "${GREEN}Running: python simple_cli.py $@${NC}"
    python simple_cli.py "$@"
fi