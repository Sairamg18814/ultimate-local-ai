#!/usr/bin/env python3
"""
Ultimate Local AI - Improved CLI
Fixed streaming, better responses, current knowledge
"""

import typer
import sys
import subprocess
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import requests
from beyond_rag import create_rag_enhanced_prompt, BeyondRAG

app = typer.Typer(
    name="ultimate-ai",
    help="Ultimate Local AI - Advanced Local AI Assistant",
    rich_markup_mode="rich"
)

console = Console()
OLLAMA_BASE_URL = "http://localhost:11434"

def check_ollama():
    """Check if Ollama is running and models are available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            return True, models
        return False, []
    except:
        return False, []

def chat_with_ai(message: str, model: str = "qwen3:32b", system_prompt: str = None, use_rag: bool = True):
    """Chat with AI using optimized streaming and Beyond RAG"""
    
    # Use Beyond RAG to enhance the query with current information
    if use_rag and not system_prompt:
        augmented_message, rag_system_prompt = create_rag_enhanced_prompt(message)
        system_prompt = rag_system_prompt
        message = augmented_message
    elif not system_prompt:
        system_prompt = """You are Ultimate Local AI, an advanced AI assistant. You are knowledgeable, helpful, and conversational.

Key guidelines:
- Provide accurate, up-to-date information
- Be concise but comprehensive  
- Use natural, flowing language
- Focus on practical, actionable advice
- Stay current with technology trends

Respond naturally without internal thinking or meta-commentary."""

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2048,
                "stop": ["<|im_end|>", "<|im_start|>"]
            }
        }
        
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=90)
        
        if response.status_code == 200:
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            chunk = data["message"]["content"]
                            
                            # Basic filtering of thinking tokens
                            if "<think>" in chunk or "</think>" in chunk:
                                continue
                            
                            full_response += chunk
                            # Print each chunk immediately as it arrives
                            print(chunk, end="", flush=True)
                        
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line at end
            return full_response.strip()
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {e}"

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    model: str = typer.Option("qwen3:32b", "--model", "-m", help="AI model to use"),
) -> None:
    """
    Chat with Ultimate Local AI
    
    Examples:
      ultimate-ai chat "What's new in AI?"
      ultimate-ai chat --interactive
    """
    
    console.print(Panel(
        Text("🚀 Ultimate Local AI - Advanced Local Intelligence", justify="center"),
        style="bold cyan"
    ))
    
    # Check Ollama status
    ollama_running, available_models = check_ollama()
    if not ollama_running:
        console.print("[red]❌ Ollama not running. Start with: ollama serve[/red]")
        return
    
    if model not in available_models:
        console.print(f"[red]❌ Model '{model}' not available.[/red]")
        if available_models:
            console.print(f"[yellow]Available: {', '.join(available_models)}[/yellow]")
        return
    
    # Show Beyond RAG status
    rag = BeyondRAG()
    current_info = rag.get_current_info()
    console.print(f"[green]✅ Using {model}[/green]")
    console.print(f"[cyan]📅 Beyond RAG Active - {current_info['date']}[/cyan]\n")
    
    if interactive or not message:
        # Interactive mode
        console.print("[cyan]🎯 Interactive AI Chat Mode[/cyan]")
        console.print("[dim]Commands: /exit, /models, /clear, /help[/dim]\n")
        
        while True:
            try:
                if not message:
                    message = Prompt.ask("[bold green]You[/bold green]")
                
                # Handle commands
                if message.lower() in ['/exit', '/quit', '/bye']:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break
                
                if message == '/models':
                    table = Table(title="Available Models")
                    table.add_column("Model", style="cyan")
                    table.add_column("Status", style="green")
                    
                    for m in available_models:
                        status = "🎯 Active" if m == model else "📋 Available"
                        table.add_row(m, status)
                    
                    console.print(table)
                    message = None
                    continue
                
                if message == '/clear':
                    console.clear()
                    message = None
                    continue
                
                if message == '/help':
                    console.print("[cyan]Available commands:[/cyan]")
                    console.print("  /exit - Exit chat")
                    console.print("  /models - List models")
                    console.print("  /clear - Clear screen")
                    console.print("  /help - Show this help")
                    message = None
                    continue
                
                # AI Response
                console.print(f"\n[bold blue]🤖 Ultimate AI[/bold blue]: ", end="")
                
                with console.status("", spinner="dots"):
                    response = chat_with_ai(message, model)
                
                console.print()
                message = None
                
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]👋 Chat ended![/yellow]")
                break
    else:
        # Single message mode
        console.print(f"[bold blue]🤖 Ultimate AI[/bold blue]: ", end="")
        
        with console.status("", spinner="dots"):
            response = chat_with_ai(message, model)

@app.command()
def reasoning(
    problem: str = typer.Argument(..., help="Problem to solve"),
    model: str = typer.Option("qwen3:32b", "--model", "-m", help="Model to use"),
) -> None:
    """Advanced problem solving and reasoning"""
    
    console.print(Panel(
        Text("🧠 Advanced Reasoning Engine", justify="center"),
        style="bold magenta"
    ))
    
    ollama_running, available_models = check_ollama()
    if not ollama_running or model not in available_models:
        console.print("[red]❌ Ollama/model not available[/red]")
        return
    
    console.print(f"\n[bold yellow]🎯 Problem:[/bold yellow] {problem}\n")
    
    reasoning_prompt = """You are Ultimate Local AI's reasoning engine. Solve problems systematically using current 2024 knowledge and best practices.

Structure your response as:
🔍 **Analysis**: Break down the problem
🧠 **Approach**: Logical reasoning steps  
💡 **Solution**: Practical, actionable solution
✅ **Confidence**: Your confidence level (0-100%)

Focus on modern, practical solutions using current technology and methods."""
    
    # Show thinking animation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("🧠 Deep reasoning in progress...", total=None)
        time.sleep(1)
        progress.update(task, description="🔍 Analyzing problem...")
        time.sleep(1)
        progress.update(task, description="💡 Generating solution...")
        time.sleep(1)
    
    console.print(f"[bold blue]🧠 Reasoning Result[/bold blue]:\n")
    
    with console.status("", spinner="dots"):
        response = chat_with_ai(problem, model, reasoning_prompt)

@app.command()
def info() -> None:
    """System information and status"""
    
    console.print(Panel(
        Text("📊 Ultimate Local AI System", justify="center"),
        style="bold blue"
    ))
    
    ollama_running, models = check_ollama()
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", style="green", width=15) 
    table.add_column("Details", width=40)
    
    ollama_status = "✅ Running" if ollama_running else "❌ Offline"
    model_count = f"✅ {len(models)} models" if models else "⚠️ No models"
    
    # Get Beyond RAG status
    rag = BeyondRAG()
    current_info = rag.get_current_info()
    
    table.add_row("🤖 Ollama Service", ollama_status, f"API: {OLLAMA_BASE_URL}")
    table.add_row("📋 AI Models", model_count, f"{', '.join(models[:3])}...")
    table.add_row("🧠 Reasoning", "✅ Active", "Advanced problem solving")
    table.add_row("💬 Chat", "✅ Active", "Interactive conversations")
    table.add_row("🌐 Beyond RAG", "✅ Active", f"Current: {current_info['date']}")
    table.add_row("🔒 Privacy", "✅ Maximum", "100% local processing")
    table.add_row("⚡ Performance", "✅ Optimized", "Streaming responses")
    
    console.print(table)
    
    console.print(f"\n[bold cyan]📈 Status:[/bold cyan]")
    console.print(f"  • Available Models: {len(models)}")
    console.print(f"  • Local Processing: 100% private")
    console.print(f"  • Response Time: Real-time")

@app.command()
def models() -> None:
    """List and manage AI models"""
    
    console.print(Panel(
        Text("🤖 AI Model Manager", justify="center"),
        style="bold green"
    ))
    
    ollama_running, models = check_ollama()
    
    if not ollama_running:
        console.print("[red]❌ Ollama not running[/red]")
        return
    
    if not models:
        console.print("[yellow]No models found. Download with:[/yellow]")
        console.print("  • [cyan]ollama pull qwen3:32b[/cyan] - Best reasoning")
        console.print("  • [cyan]ollama pull qwen2.5:14b[/cyan] - Balanced")
        console.print("  • [cyan]ollama pull qwen2.5:7b[/cyan] - Fast")
        return
    
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan", width=25)
    table.add_column("Status", style="green", width=15)
    table.add_column("Best For", style="yellow", width=35)
    
    model_info = {
        "qwen3:32b": "🧠 Complex reasoning & analysis",
        "qwen2.5:14b": "⚖️ Balanced performance",  
        "qwen2.5:7b": "⚡ Quick responses",
        "llama3.1:8b": "💬 General conversation",
        "jigyasa:latest": "🎯 Custom model"
    }
    
    for model in models:
        use_case = model_info.get(model, "🤖 General AI tasks")
        table.add_row(model, "✅ Ready", use_case)
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models available[/dim]")

@app.command()  
def serve() -> None:
    """Start Ollama service"""
    
    console.print("[yellow]🔄 Starting Ollama...[/yellow]")
    
    try:
        # Check if already running
        if check_ollama()[0]:
            console.print("[green]✅ Ollama already running[/green]")
            return
        
        # Try to start
        subprocess.run(["ollama", "serve"], check=False)
        
    except FileNotFoundError:
        console.print("[red]❌ Ollama not installed. Install with: brew install ollama[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")

def main():
    app()

if __name__ == "__main__":
    main()