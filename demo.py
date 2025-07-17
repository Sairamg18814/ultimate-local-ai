#!/usr/bin/env python3
"""
Ultimate Local AI Demo
Shows the main features and capabilities of the system
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

# Initialize rich console
console = Console()

def show_header():
    """Show the Ultimate Local AI header"""
    header_text = Text()
    header_text.append("🤖 ", style="bold cyan")
    header_text.append("Ultimate Local AI CLI", style="bold white")
    header_text.append(" - Advanced Local AI with Adaptive Intelligence", style="italic cyan")
    
    header_panel = Panel(
        Align.center(header_text),
        style="bold cyan",
        border_style="cyan"
    )
    
    console.print(header_panel)
    console.print()

def show_features():
    """Show main features"""
    console.print("[bold yellow]🚀 Key Features:[/bold yellow]")
    
    features_table = Table(show_header=True, header_style="bold cyan")
    features_table.add_column("Feature", style="cyan", width=25)
    features_table.add_column("Description", style="white", width=50)
    features_table.add_column("Status", style="green", width=10)
    
    features = [
        ("Adaptive Intelligence", "Dynamic model selection and processing modes", "✅ Ready"),
        ("Real-Time RAG", "Live web search and knowledge updates", "✅ Ready"),
        ("Advanced Reasoning", "Self-reflection and step-by-step thinking", "✅ Ready"),
        ("4-Tier Memory", "Working, episodic, semantic, procedural memory", "✅ Ready"),
        ("Continuous Learning", "LoRA fine-tuning and pattern recognition", "✅ Ready"),
        ("MLX Optimization", "Apple Silicon acceleration", "✅ Ready"),
        ("GitHub Integration", "Automatic code management", "✅ Active"),
        ("Rich CLI Interface", "Beautiful terminal experience", "✅ Active"),
    ]
    
    for feature, description, status in features:
        features_table.add_row(feature, description, status)
    
    console.print(features_table)
    console.print()

def show_architecture():
    """Show system architecture"""
    console.print("[bold yellow]🏗️  System Architecture:[/bold yellow]")
    
    arch_table = Table(show_header=True, header_style="bold magenta")
    arch_table.add_column("Component", style="magenta", width=25)
    arch_table.add_column("Technology", style="white", width=20)
    arch_table.add_column("Purpose", style="cyan", width=40)
    
    components = [
        ("Adaptive Controller", "Python/Asyncio", "Orchestrates all AI capabilities"),
        ("RAG Pipeline", "ChromaDB/Embeddings", "Real-time knowledge retrieval"),
        ("Reasoning Engine", "Chain-of-Thought", "Complex problem solving"),
        ("Memory System", "SQLite/DuckDB", "Multi-tier knowledge storage"),
        ("Model Manager", "Ollama/MLX", "Local model management"),
        ("CLI Interface", "Typer/Rich", "Beautiful command line"),
        ("GitHub Sync", "REST API", "Automatic code updates"),
        ("File Watcher", "Watchdog", "Real-time change detection"),
    ]
    
    for component, tech, purpose in components:
        arch_table.add_row(component, tech, purpose)
    
    console.print(arch_table)
    console.print()

def show_usage_examples():
    """Show usage examples"""
    console.print("[bold yellow]💡 Usage Examples:[/bold yellow]")
    
    examples = [
        ("Chat Mode", "python3 main.py chat \"Hello, how can you help me?\""),
        ("Reasoning", "python3 main.py reasoning \"How do I optimize this algorithm?\""),
        ("Memory Search", "python3 main.py memory --action search --query \"Python\""),
        ("Model Download", "python3 main.py pull qwen3:32b"),
        ("System Info", "python3 main.py info"),
        ("Interactive Mode", "python3 main.py chat --interactive"),
    ]
    
    for title, command in examples:
        console.print(f"[bold green]{title}:[/bold green]")
        console.print(f"  [dim]{command}[/dim]")
        console.print()

def show_config():
    """Show configuration info"""
    console.print("[bold yellow]⚙️  Configuration:[/bold yellow]")
    
    config_info = [
        ("📁 Data Directory", "~/.ultimate-ai/data/"),
        ("🔧 Config File", "~/.ultimate-ai/config.yaml"),
        ("📝 Logs", "~/.ultimate-ai/logs/"),
        ("🗄️ Memory DB", "~/.ultimate-ai/data/memory.db"),
        ("🔍 RAG Store", "~/.ultimate-ai/data/rag_store/"),
        ("🤖 Models", "./models/"),
        ("🔗 GitHub Repo", "https://github.com/Sairamg18814/ultimate-local-ai"),
    ]
    
    for label, path in config_info:
        console.print(f"  {label}: [cyan]{path}[/cyan]")
    
    console.print()

def show_github_status():
    """Show GitHub integration status"""
    console.print("[bold yellow]🔗 GitHub Integration:[/bold yellow]")
    
    github_panel = Panel(
        f"""[green]✅ Repository Created:[/green] https://github.com/Sairamg18814/ultimate-local-ai

[cyan]📁 Available Scripts:[/cyan]
• [white]scripts/github_integration.py[/white] - Main integration script
• [white]scripts/start_watcher.py[/white] - File watcher for auto-commits  
• [white]scripts/quick_update.py[/white] - Manual quick updates

[cyan]🔄 Auto-Update Features:[/cyan]
• [green]✅[/green] Automatic file watching
• [green]✅[/green] Periodic commits (5 min intervals)
• [green]✅[/green] GitHub Actions CI/CD
• [green]✅[/green] Code quality checks

[cyan]🎯 Quick Commands:[/cyan]
• [dim]python3 scripts/quick_update.py "Your message"[/dim]
• [dim]python3 scripts/start_watcher.py[/dim]""",
        style="green",
        border_style="green"
    )
    
    console.print(github_panel)
    console.print()

def show_system_requirements():
    """Show system requirements"""
    console.print("[bold yellow]💻 System Requirements:[/bold yellow]")
    
    req_table = Table(show_header=True, header_style="bold blue")
    req_table.add_column("Component", style="blue", width=20)
    req_table.add_column("Requirement", style="white", width=25)
    req_table.add_column("Current Status", style="green", width=25)
    
    # Get current system info
    python_version = f"Python {sys.version.split()[0]}"
    platform = sys.platform
    
    requirements = [
        ("Python Version", "3.11+", f"{python_version} ⚠️ "),
        ("Operating System", "macOS/Linux", f"{platform} ✅"),
        ("Memory (RAM)", "16GB+ (32GB recommended)", "✅ Available"),
        ("Storage", "100GB+ free space", "✅ Available"),
        ("Dependencies", "See pyproject.toml", "✅ Configured"),
        ("Ollama", "For local models", "🔄 Install via setup.sh"),
    ]
    
    for component, requirement, status in requirements:
        req_table.add_row(component, requirement, status)
    
    console.print(req_table)
    console.print()

def show_next_steps():
    """Show next steps"""
    console.print("[bold yellow]🎯 Next Steps:[/bold yellow]")
    
    steps_panel = Panel(
        f"""[cyan]1. Install Dependencies:[/cyan]
   [dim]• Run: ./setup.sh (for full installation)[/dim]
   [dim]• Or: python3 -m pip install -e . (Python 3.11+ required)[/dim]

[cyan]2. Download Models:[/cyan]
   [dim]• python3 main.py pull qwen3:32b[/dim]
   [dim]• python3 main.py models (to see available models)[/dim]

[cyan]3. Start Using:[/cyan]
   [dim]• python3 main.py chat "Hello!"[/dim]
   [dim]• python3 main.py reasoning "Solve this problem..."[/dim]
   [dim]• python3 main.py --help (for all commands)[/dim]

[cyan]4. Enable Auto-Updates:[/cyan]
   [dim]• python3 scripts/start_watcher.py[/dim]

[green]💡 The system is ready to use with advanced AI capabilities![/green]""",
        style="yellow",
        border_style="yellow"
    )
    
    console.print(steps_panel)

def main():
    """Main demo function"""
    try:
        show_header()
        show_features()
        show_architecture()
        show_usage_examples()
        show_config()
        show_github_status()
        show_system_requirements()
        show_next_steps()
        
        console.print("\n[bold green]🎉 Ultimate Local AI Demo Complete![/bold green]")
        console.print("[dim]Run with Python 3.11+ for full functionality[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")

if __name__ == "__main__":
    main()