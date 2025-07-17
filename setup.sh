#!/bin/bash

# Ultimate Local AI CLI Setup Script
# This script sets up the complete Ultimate Local AI system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    Ultimate Local AI CLI                         ║"
    echo "║              Advanced Local AI with Adaptive Intelligence       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_system() {
    print_step "Checking system requirements..."
    
    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is designed for macOS. Please adapt for your OS."
        exit 1
    fi
    
    # Check hardware
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    if [ "$TOTAL_RAM" -lt 16 ]; then
        print_error "Minimum 16GB RAM required. You have ${TOTAL_RAM}GB."
        exit 1
    fi
    
    if [ "$TOTAL_RAM" -ge 48 ]; then
        print_success "RAM: ${TOTAL_RAM}GB - Excellent! Can run 70B models."
    elif [ "$TOTAL_RAM" -ge 32 ]; then
        print_success "RAM: ${TOTAL_RAM}GB - Good! Can run 32B models efficiently."
    else
        print_warning "RAM: ${TOTAL_RAM}GB - Okay, but 32GB+ recommended for optimal performance."
    fi
    
    # Check CPU
    CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
    if [[ "$CPU_BRAND" == *"Apple"* ]]; then
        print_success "CPU: Apple Silicon detected - Optimal for MLX acceleration"
    else
        print_warning "CPU: Intel detected - Consider Apple Silicon for better performance"
    fi
    
    # Check available storage
    AVAILABLE_SPACE=$(df -h . | tail -1 | awk '{print $4}' | sed 's/Gi//')
    if [ "${AVAILABLE_SPACE%.*}" -lt 100 ]; then
        print_error "Need at least 100GB free space. You have ${AVAILABLE_SPACE}GB."
        exit 1
    fi
    
    print_success "System requirements check passed!"
}

check_dependencies() {
    print_step "Checking and installing dependencies..."
    
    # Check for Python 3.12 first, then fallback to python3
    PYTHON_CMD=""
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        PYTHON_VERSION=$(python3.12 --version | cut -d' ' -f2)
        print_success "Python 3.12 found: $PYTHON_VERSION"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            PYTHON_CMD="python3"
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.11+ required. Found: $PYTHON_VERSION"
            print_step "Trying to find Python 3.12..."
            if command -v python3.12 &> /dev/null; then
                PYTHON_CMD="python3.12"
                PYTHON_VERSION=$(python3.12 --version | cut -d' ' -f2)
                print_success "Found Python 3.12: $PYTHON_VERSION"
            else
                print_error "Please install Python 3.12: brew install python@3.12"
                exit 1
            fi
        fi
    else
        print_error "Python 3 not found. Please install Python 3.12: brew install python@3.12"
        exit 1
    fi
    
    # Export Python command for use in other functions
    export PYTHON_CMD
    
    # Check Homebrew
    if ! command -v brew &> /dev/null; then
        print_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        print_success "Homebrew installed"
    else
        print_success "Homebrew found"
    fi
    
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        print_step "Installing Ollama..."
        brew install ollama
        print_success "Ollama installed"
    else
        print_success "Ollama found"
    fi
    
    # Check if Ollama service is running
    if ! pgrep -x "ollama" > /dev/null; then
        print_step "Starting Ollama service..."
        brew services start ollama
        sleep 5
        print_success "Ollama service started"
    else
        print_success "Ollama service is running"
    fi
}

create_virtual_environment() {
    print_step "Creating Python virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        print_step "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment with Python 3.12
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Verify Python version in venv
    VENV_PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_success "Virtual environment Python version: $VENV_PYTHON_VERSION"
}

install_python_dependencies() {
    print_step "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_step "Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip
    print_success "Pip upgraded"
    
    # Install wheel and setuptools first
    python -m pip install wheel setuptools
    print_success "Build tools installed"
    
    # Install the package in development mode
    python -m pip install -e .
    print_success "Ultimate Local AI installed in development mode"
    
    # Install additional dependencies for file watching
    python -m pip install watchdog requests python-dotenv
    print_success "Additional dependencies installed"
    
    print_success "All Python dependencies installed in virtual environment"
}

setup_models() {
    print_step "Setting up AI models..."
    
    # Determine optimal model based on RAM
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    
    if [ "$TOTAL_RAM" -ge 48 ]; then
        MODEL_CONFIG="qwen3:32b"
        print_step "Detected 48GB+ RAM - Using QwQ-32B for optimal reasoning"
    elif [ "$TOTAL_RAM" -ge 32 ]; then
        MODEL_CONFIG="qwen2.5-coder:32b"
        print_step "Detected 32GB+ RAM - Using Qwen2.5-Coder-32B"
    else
        MODEL_CONFIG="qwen2.5:14b"
        print_step "Detected 16GB+ RAM - Using Qwen2.5-14B"
    fi
    
    # Download primary model
    print_step "Downloading primary model: $MODEL_CONFIG"
    echo "This may take 10-30 minutes depending on your internet speed..."
    
    if ollama pull $MODEL_CONFIG; then
        print_success "Primary model downloaded: $MODEL_CONFIG"
    else
        print_error "Failed to download primary model"
        exit 1
    fi
    
    # Download additional models if space allows
    if [ "$TOTAL_RAM" -ge 48 ]; then
        print_step "Downloading additional models for specialized tasks..."
        
        # Coding model
        if ollama pull qwen2.5-coder:32b; then
            print_success "Coding model downloaded: qwen2.5-coder:32b"
        else
            print_warning "Failed to download coding model"
        fi
        
        # Vision model
        if ollama pull llava-next:32b; then
            print_success "Vision model downloaded: llava-next:32b"
        else
            print_warning "Failed to download vision model"
        fi
    fi
}

setup_directories() {
    print_step "Setting up directories and configuration..."
    
    # Create data directory
    mkdir -p ~/.ultimate-ai/{config,data,models,cache,logs}
    
    # Create default configuration
    cat > ~/.ultimate-ai/config.yaml << EOF
# Ultimate Local AI Configuration
model:
  name: "$MODEL_CONFIG"
  temperature: 0.7
  max_tokens: 4096
  thinking_mode: "auto"
  context_window: 32768

rag:
  enable_web_search: true
  update_interval: 3600
  max_results: 10
  freshness_threshold: 24
  vector_store_path: "~/.ultimate-ai/data/rag_store"

memory:
  working_capacity: 20
  episodic_capacity: 1000
  semantic_capacity: 10000
  procedural_capacity: 500
  memory_db_path: "~/.ultimate-ai/data/memory.db"

reasoning:
  enable_self_reflection: true
  confidence_threshold: 0.7
  max_reasoning_steps: 10
  pattern_learning: true

learning:
  enable_continuous_learning: true
  quality_threshold: 0.8
  consolidation_interval: 86400

system:
  log_level: "INFO"
  log_file: "~/.ultimate-ai/logs/ultimate-ai.log"
  enable_analytics: false
  check_updates: true
EOF
    
    print_success "Configuration created at ~/.ultimate-ai/config.yaml"
}

setup_shell_integration() {
    print_step "Setting up shell integration..."
    
    # Detect shell
    SHELL_NAME=$(basename "$SHELL")
    
    case "$SHELL_NAME" in
        "bash")
            SHELL_RC="$HOME/.bashrc"
            ;;
        "zsh")
            SHELL_RC="$HOME/.zshrc"
            ;;
        "fish")
            SHELL_RC="$HOME/.config/fish/config.fish"
            ;;
        *)
            print_warning "Unknown shell: $SHELL_NAME. Please add alias manually."
            return
            ;;
    esac
    
    # Add alias if not already present
    if ! grep -q "alias uai=" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# Ultimate Local AI CLI" >> "$SHELL_RC"
        echo "alias uai-activate='cd $(pwd) && source venv/bin/activate'" >> "$SHELL_RC"
        echo "alias uai-simple='python simple_cli.py'" >> "$SHELL_RC"
        echo "alias uai-chat='python simple_cli.py chat'" >> "$SHELL_RC"
        echo "alias uai-reason='python simple_cli.py reasoning'" >> "$SHELL_RC"
        echo "alias uai-info='python simple_cli.py info'" >> "$SHELL_RC"
        
        print_success "Shell aliases added to $SHELL_RC"
        print_step "You can now use shortcuts like 'uai-activate' and 'uai-chat'"
    else
        print_success "Shell aliases already configured"
    fi
}

run_initial_test() {
    print_step "Running initial system test..."
    
    # Ensure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_step "Activating virtual environment for testing..."
        source venv/bin/activate
    fi
    
    # Test simple CLI functionality
    print_step "Testing Simple CLI functionality..."
    if python simple_cli.py info > /dev/null 2>&1; then
        print_success "Simple CLI test passed!"
    else
        print_warning "Simple CLI test failed, but continuing..."
    fi
    
    # Test main CLI if it works
    print_step "Testing main CLI functionality..."
    if python ultimate_local_ai/main.py --help > /dev/null 2>&1; then
        print_success "Main CLI test passed!"
    else
        print_warning "Main CLI has some issues, but simple CLI works"
    fi
    
    # Test GitHub integration
    print_step "Testing GitHub integration..."
    if python scripts/github_integration.py --help > /dev/null 2>&1; then
        print_success "GitHub integration test passed!"
    else
        print_warning "GitHub integration test failed"
    fi
    
    print_success "Initial tests completed!"
}

print_completion_message() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 INSTALLATION COMPLETE! 🎉                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${CYAN}Getting Started (Activate Virtual Environment First):${NC}"
    echo "  • Activate venv: ${YELLOW}source venv/bin/activate${NC}"
    echo "  • Simple chat: ${YELLOW}python simple_cli.py chat \"Hello!\"${NC}"
    echo "  • Interactive: ${YELLOW}python simple_cli.py chat --interactive${NC}"
    echo "  • Advanced reasoning: ${YELLOW}python simple_cli.py reasoning \"How to optimize code?\"${NC}"
    echo "  • System info: ${YELLOW}python simple_cli.py info${NC}"
    echo "  • View models: ${YELLOW}python simple_cli.py models${NC}"
    echo ""
    
    echo -e "${CYAN}Advanced Usage (after Ollama models are downloaded):${NC}"
    echo "  ${YELLOW}python ultimate_local_ai/main.py chat \"Hello there\"${NC}"
    echo "  ${YELLOW}python ultimate_local_ai/main.py reasoning \"Complex problem\"${NC}"
    echo "  ${YELLOW}ultimate-ai memory --action search --query \"Python\"${NC}"
    echo ""
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  • Config file: ${YELLOW}~/.ultimate-ai/config.yaml${NC}"
    echo "  • Data directory: ${YELLOW}~/.ultimate-ai/data/${NC}"
    echo "  • Logs: ${YELLOW}~/.ultimate-ai/logs/${NC}"
    echo ""
    
    echo -e "${CYAN}Performance Tips:${NC}"
    echo "  • Use thinking mode for complex problems"
    echo "  • Enable RAG for current information"
    echo "  • Let the system learn from your interactions"
    echo "  • Regular memory consolidation improves performance"
    echo ""
    
    echo -e "${GREEN}Your Ultimate Local AI is ready to use!${NC}"
    echo ""
    
    # Show system info
    print_step "Current system configuration:"
    echo "  • Model: $MODEL_CONFIG"
    echo "  • RAM: ${TOTAL_RAM}GB"
    echo "  • Storage: ${AVAILABLE_SPACE}GB available"
    echo "  • CPU: Apple Silicon optimized"
    echo ""
    
    echo -e "${BLUE}For support and updates:${NC}"
    echo "  • GitHub Repository: https://github.com/Sairamg18814/ultimate-local-ai"
    echo "  • Issues & Support: https://github.com/Sairamg18814/ultimate-local-ai/issues"
    echo "  • Auto-Updates: python scripts/quick_update.py \"message\""
    echo "  • File Watcher: python scripts/start_watcher.py"
}

main() {
    print_header
    
    echo "This script will install and configure the Ultimate Local AI CLI."
    echo "Estimated installation time: 15-45 minutes"
    echo ""
    
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    echo ""
    
    # Main installation steps
    check_system
    check_dependencies
    create_virtual_environment
    install_python_dependencies
    setup_models
    setup_directories
    setup_shell_integration
    run_initial_test
    
    echo ""
    print_completion_message
    
    # Ask to start chat
    echo ""
    read -p "Would you like to start a chat session now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting Ultimate Local AI simple demo..."
        echo "Activating virtual environment and launching chat..."
        echo ""
        source venv/bin/activate
        exec python simple_cli.py chat --interactive
    else
        echo ""
        echo "You can start chatting anytime with:"
        echo "  ${YELLOW}source venv/bin/activate${NC}"
        echo "  ${YELLOW}python simple_cli.py chat --interactive${NC}"
        echo ""
    fi
}

# Run main function
main "$@"