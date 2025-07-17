"""Ultimate Local AI CLI - Main entry point."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from .core.adaptive_controller import AdaptiveIntelligenceController
from .interface.rich_interface import RichInterface
from .utils.config_manager import ConfigManager
from .utils.system_check import SystemChecker

# Initialize CLI app
app = typer.Typer(
    name="ultimate-ai",
    help="Ultimate Local AI CLI with adaptive intelligence, real-time RAG, and continuous learning",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global instances
console = Console()
controller: Optional[AdaptiveIntelligenceController] = None
config: Optional[ConfigManager] = None
interface: Optional[RichInterface] = None


def setup_logging() -> None:
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ultimate_ai.log")
        ]
    )


async def initialize_system() -> None:
    """Initialize the system components."""
    global controller, config, interface
    
    try:
        # Setup logging
        setup_logging()
        
        # Initialize config manager
        config = ConfigManager()
        await config.initialize()
        
        # Initialize rich interface
        interface = RichInterface(console, config)
        
        # System check
        system_checker = SystemChecker()
        system_info = await system_checker.check_system()
        
        if not system_info["compatible"]:
            console.print("[red]System compatibility check failed![/red]")
            for issue in system_info["issues"]:
                console.print(f"[red]- {issue}[/red]")
            raise typer.Exit(1)
        
        # Initialize adaptive controller
        controller = AdaptiveIntelligenceController(config.get_config())
        
        # Show initialization progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            init_task = progress.add_task("Initializing Ultimate Local AI...", total=4)
            
            # Initialize controller
            progress.update(init_task, description="Initializing AI Controller...")
            await controller.initialize()
            progress.advance(init_task)
            
            # Initialize interface
            progress.update(init_task, description="Setting up interface...")
            await interface.initialize()
            progress.advance(init_task)
            
            # Load models
            progress.update(init_task, description="Loading AI models...")
            await controller.model_manager.ensure_models_loaded()
            progress.advance(init_task)
            
            # Final setup
            progress.update(init_task, description="Finalizing setup...")
            await asyncio.sleep(0.5)  # Brief pause for UI
            progress.advance(init_task)
        
        console.print("[green]✅ Ultimate Local AI initialized successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        raise


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Direct message to send"),
    model: str = typer.Option("qwen3:32b", "--model", "-m", help="Model to use"),
    thinking: bool = typer.Option(None, "--thinking/--no-thinking", help="Force thinking mode"),
    reasoning: bool = typer.Option(True, "--reasoning/--no-reasoning", help="Enable reasoning"),
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Enable RAG"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable memory"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature (0-1)"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save conversation"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream responses"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Start a conversation with the AI assistant.
    
    This is the main command for interacting with the Ultimate Local AI.
    It supports both single message mode and interactive chat mode.
    """
    try:
        # Initialize system
        asyncio.run(initialize_system())
        
        # Configure settings
        settings = {
            "model": model,
            "thinking_mode": thinking,
            "enable_reasoning": reasoning,
            "enable_rag": rag,
            "enable_memory": memory,
            "temperature": temperature,
            "save_conversation": save,
            "stream_responses": stream,
            "verbose": verbose
        }
        
        # Update controller settings
        controller.update_settings(settings)
        
        if message:
            # Single message mode
            asyncio.run(handle_single_message(message, settings))
        else:
            # Interactive chat mode
            asyncio.run(interactive_chat(settings))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


async def handle_single_message(message: str, settings: dict) -> None:
    """Handle single message mode."""
    try:
        # Show processing indicator
        with Live(interface.create_thinking_indicator(), refresh_per_second=4) as live:
            # Process the message
            result = await controller.process_query(
                query=message,
                user_preferences=settings
            )
            
            # Update display with result
            live.update(interface.create_response_display(result))
            
            # Wait a moment for user to read
            await asyncio.sleep(2)
        
        # Show final result
        interface.display_response(result)
        
    except Exception as e:
        console.print(f"[red]Error processing message: {e}[/red]")
        raise


async def interactive_chat(settings: dict) -> None:
    """Handle interactive chat mode."""
    try:
        # Show welcome message
        interface.show_welcome_message()
        
        # Start conversation loop
        conversation_id = f"conversation_{datetime.now().timestamp()}"
        
        while True:
            try:
                # Get user input
                user_input = interface.get_user_input()
                
                # Check for commands
                if user_input.strip().lower() in ["/exit", "/quit", "/q"]:
                    break
                elif user_input.strip().lower() in ["/help", "/h"]:
                    interface.show_help()
                    continue
                elif user_input.strip().lower() in ["/clear", "/cls"]:
                    console.clear()
                    continue
                elif user_input.strip().lower() in ["/stats", "/status"]:
                    await show_system_stats()
                    continue
                elif user_input.strip().lower() in ["/memory", "/mem"]:
                    await show_memory_stats()
                    continue
                elif user_input.strip().lower().startswith("/config"):
                    await handle_config_command(user_input)
                    continue
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Process the message
                await process_interactive_message(user_input, conversation_id, settings)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        # Farewell message
        interface.show_farewell_message()
        
    except Exception as e:
        console.print(f"[red]Error in interactive chat: {e}[/red]")
        raise


async def process_interactive_message(message: str, conversation_id: str, settings: dict) -> None:
    """Process a message in interactive mode."""
    try:
        # Show processing indicator
        with Live(interface.create_thinking_indicator(), refresh_per_second=4) as live:
            
            # Process the message
            result = await controller.process_query(
                query=message,
                context={"conversation_id": conversation_id},
                user_preferences=settings
            )
            
            # Update display based on processing mode
            if result.mode_used.value == "thinking":
                live.update(interface.create_thinking_display(result))
            else:
                live.update(interface.create_response_display(result))
            
            # Brief pause for smooth transition
            await asyncio.sleep(0.5)
        
        # Display final response
        interface.display_response(result)
        
        # Show additional info if verbose
        if settings.get("verbose"):
            interface.show_processing_details(result)
        
    except Exception as e:
        console.print(f"[red]Error processing message: {e}[/red]")
        raise


@app.command()
def models() -> None:
    """List available models and their status."""
    try:
        asyncio.run(initialize_system())
        
        # Get model information
        models_info = asyncio.run(controller.model_manager.get_models_info())
        
        # Display models table
        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Performance", style="magenta")
        table.add_column("Recommended Use", style="blue")
        
        for model_info in models_info:
            table.add_row(
                model_info["name"],
                model_info["status"],
                model_info["size"],
                model_info["performance"],
                model_info["recommended_use"]
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        sys.exit(1)


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model name to download"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download and install a model."""
    try:
        asyncio.run(initialize_system())
        
        # Download model with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            download_task = progress.add_task(f"Downloading {model}...", total=None)
            
            async def progress_callback(percent, status):
                progress.update(download_task, description=f"Downloading {model}: {status}")
            
            # Download the model
            await controller.model_manager.download_model(model, force, progress_callback)
            
            progress.update(download_task, description=f"✅ {model} downloaded successfully")
        
        console.print(f"[green]Model {model} is ready to use![/green]")
        
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        sys.exit(1)


@app.command()
def reasoning(
    problem: str = typer.Argument(..., help="Problem to solve"),
    method: str = typer.Option("auto", "--method", "-m", help="Reasoning method"),
    show_steps: bool = typer.Option(True, "--steps/--no-steps", help="Show reasoning steps"),
    show_confidence: bool = typer.Option(True, "--confidence/--no-confidence", help="Show confidence"),
    save_pattern: bool = typer.Option(True, "--save-pattern/--no-save-pattern", help="Save successful patterns"),
) -> None:
    """Solve a problem using advanced reasoning capabilities."""
    try:
        asyncio.run(initialize_system())
        
        # Use reasoning-focused settings
        settings = {
            "enable_reasoning": True,
            "thinking_mode": True,
            "enable_rag": True,
            "enable_memory": True,
            "temperature": 0.7,
            "verbose": True
        }
        
        # Process with reasoning
        with Live(interface.create_thinking_indicator(), refresh_per_second=4) as live:
            result = await controller.process_query(
                query=problem,
                context={"reasoning_method": method},
                user_preferences=settings
            )
            
            live.update(interface.create_reasoning_display(result))
            await asyncio.sleep(1)
        
        # Display results
        interface.display_reasoning_result(result, show_steps, show_confidence)
        
        # Save pattern if successful
        if save_pattern and result.confidence > 0.8:
            await controller.reasoning_engine.learn_pattern(
                problem, result.thinking_process, result.confidence
            )
            console.print("[green]✅ Reasoning pattern saved for future use[/green]")
        
    except Exception as e:
        console.print(f"[red]Error in reasoning: {e}[/red]")
        sys.exit(1)


@app.command()
def memory(
    action: str = typer.Option("stats", "--action", "-a", help="Action: stats, search, clear"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
) -> None:
    """Manage memory system."""
    try:
        asyncio.run(initialize_system())
        
        if action == "stats":
            await show_memory_stats()
        elif action == "search" and query:
            await search_memory(query, limit)
        elif action == "clear":
            await clear_memory()
        else:
            console.print("[red]Invalid action or missing query[/red]")
            
    except Exception as e:
        console.print(f"[red]Error managing memory: {e}[/red]")
        sys.exit(1)


@app.command()
def config(
    action: str = typer.Option("show", "--action", "-a", help="Action: show, set, reset"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Configuration key"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="Configuration value"),
) -> None:
    """Manage configuration."""
    try:
        asyncio.run(initialize_system())
        
        if action == "show":
            interface.show_config(config.get_config())
        elif action == "set" and key and value:
            config.set_config(key, value)
            console.print(f"[green]Configuration updated: {key} = {value}[/green]")
        elif action == "reset":
            config.reset_config()
            console.print("[green]Configuration reset to defaults[/green]")
        else:
            console.print("[red]Invalid action or missing parameters[/red]")
            
    except Exception as e:
        console.print(f"[red]Error managing config: {e}[/red]")
        sys.exit(1)


@app.command()
def info() -> None:
    """Show system information and status."""
    try:
        asyncio.run(initialize_system())
        
        # System information
        system_checker = SystemChecker()
        system_info = await system_checker.get_detailed_info()
        
        interface.show_system_info(system_info)
        
        # Performance stats
        if controller:
            performance_stats = controller.get_performance_stats()
            interface.show_performance_stats(performance_stats)
        
    except Exception as e:
        console.print(f"[red]Error getting system info: {e}[/red]")
        sys.exit(1)


@app.command()
def benchmark(
    test_type: str = typer.Option("quick", "--type", "-t", help="Test type: quick, full, reasoning"),
    iterations: int = typer.Option(10, "--iterations", "-i", help="Number of iterations"),
) -> None:
    """Run performance benchmarks."""
    try:
        asyncio.run(initialize_system())
        
        console.print(f"[cyan]Running {test_type} benchmark with {iterations} iterations...[/cyan]")
        
        # Run benchmark
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            benchmark_task = progress.add_task("Running benchmark...", total=iterations)
            
            results = []
            
            for i in range(iterations):
                # Generate test query
                test_query = f"What is the result of {i} + {i}?"
                
                # Process query
                start_time = datetime.now()
                result = await controller.process_query(test_query)
                end_time = datetime.now()
                
                # Record result
                results.append({
                    "iteration": i + 1,
                    "query": test_query,
                    "processing_time": (end_time - start_time).total_seconds(),
                    "confidence": result.confidence,
                    "mode_used": result.mode_used.value,
                    "tokens_used": result.tokens_used
                })
                
                progress.advance(benchmark_task)
        
        # Display results
        interface.show_benchmark_results(results)
        
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        sys.exit(1)


async def show_system_stats() -> None:
    """Show system statistics."""
    try:
        stats = {}
        
        if controller:
            stats["controller"] = controller.get_performance_stats()
            stats["memory"] = controller.memory_system.get_memory_stats()
            stats["rag"] = controller.rag_pipeline.get_statistics()
            stats["reasoning"] = controller.reasoning_engine.get_performance_stats()
        
        interface.show_system_stats(stats)
        
    except Exception as e:
        console.print(f"[red]Error getting system stats: {e}[/red]")


async def show_memory_stats() -> None:
    """Show memory statistics."""
    try:
        if controller:
            memory_stats = controller.memory_system.get_memory_stats()
            interface.show_memory_stats(memory_stats)
        
    except Exception as e:
        console.print(f"[red]Error getting memory stats: {e}[/red]")


async def search_memory(query: str, limit: int) -> None:
    """Search memory system."""
    try:
        if controller:
            results = await controller.memory_system.retrieve_relevant(query, limit)
            interface.show_memory_search_results(results)
        
    except Exception as e:
        console.print(f"[red]Error searching memory: {e}[/red]")


async def clear_memory() -> None:
    """Clear memory system."""
    try:
        if controller:
            # Ask for confirmation
            if typer.confirm("Are you sure you want to clear all memory?"):
                await controller.memory_system.clear_all_memory()
                console.print("[green]Memory cleared successfully[/green]")
            else:
                console.print("[yellow]Memory clear cancelled[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error clearing memory: {e}[/red]")


async def handle_config_command(command: str) -> None:
    """Handle configuration commands."""
    try:
        parts = command.split()
        
        if len(parts) == 1:
            # Show config
            interface.show_config(config.get_config())
        elif len(parts) == 3:
            # Set config
            key, value = parts[1], parts[2]
            config.set_config(key, value)
            console.print(f"[green]Configuration updated: {key} = {value}[/green]")
        else:
            console.print("[yellow]Usage: /config [key] [value][/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error handling config command: {e}[/red]")


def main() -> None:
    """Main entry point."""
    try:
        # Show welcome banner if no arguments
        if len(sys.argv) == 1:
            console.print(Panel(
                Text("Ultimate Local AI CLI", style="bold cyan", justify="center") + "\n\n" +
                Text("🚀 Adaptive Intelligence • 🧠 Advanced Reasoning • 📚 Real-Time RAG", justify="center") + "\n" +
                Text("🔄 Continuous Learning • 💾 Integrated Memory • 🔒 Complete Privacy", justify="center") + "\n\n" +
                Text("Use 'ultimate-ai chat' to start a conversation", style="dim", justify="center") + "\n" +
                Text("Use 'ultimate-ai --help' for available commands", style="dim", justify="center"),
                title="🤖 Welcome to Ultimate Local AI",
                border_style="cyan",
                padding=(1, 2)
            ))
            return
        
        # Run CLI
        app()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()