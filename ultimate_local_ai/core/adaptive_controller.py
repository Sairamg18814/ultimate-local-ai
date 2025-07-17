"""Adaptive Intelligence Controller - Core orchestrator for the Ultimate Local AI CLI."""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..rag.realtime_rag import RealTimeRAG
from ..memory.integrated_memory import IntegratedMemorySystem
from ..reasoning.advanced_reasoning import AdvancedReasoningEngine
from ..models.model_manager import ModelManager
from ..utils.query_classifier import QueryClassifier
from ..utils.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    SIMPLE_FACT = "simple_fact"
    CURRENT_INFO = "current_info"
    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE = "creative"
    PERSONAL_CONTEXT = "personal_context"
    MULTI_MODAL = "multi_modal"
    SYSTEM_COMMAND = "system_command"


class ProcessingMode(Enum):
    """Processing modes for different query types."""
    THINKING = "thinking"
    NON_THINKING = "non_thinking"
    HYBRID = "hybrid"


class QueryAnalysis(BaseModel):
    """Analysis result for a query."""
    query_type: QueryType
    complexity: float = Field(ge=0.0, le=1.0, description="Complexity score 0-1")
    needs_current_info: bool = False
    needs_reasoning: bool = False
    needs_tools: bool = False
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of processing a query."""
    response: str
    thinking_process: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in response")
    processing_time: float
    tokens_used: int
    mode_used: ProcessingMode
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdaptiveIntelligenceController:
    """
    Main controller that orchestrates all AI capabilities.
    
    This is the brain of the system that decides how to process each query
    by selecting the appropriate combination of:
    - Processing mode (thinking vs non-thinking)
    - RAG retrieval strategy
    - Memory systems
    - Reasoning engines
    - Tools and capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize core components
        self.model_manager = ModelManager(config)
        self.rag_pipeline = RealTimeRAG(config)
        self.memory_system = IntegratedMemorySystem(config)
        self.reasoning_engine = AdvancedReasoningEngine(config)
        self.query_classifier = QueryClassifier(config)
        self.context_builder = ContextBuilder(config)
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        self.mode_usage_stats = {mode: 0 for mode in ProcessingMode}
        
        # Adaptation parameters
        self.complexity_threshold = config.get("complexity_threshold", 0.6)
        self.thinking_mode_bias = config.get("thinking_mode_bias", 0.1)
        self.adaptive_learning_rate = config.get("adaptive_learning_rate", 0.01)
        
        logger.info("Adaptive Intelligence Controller initialized")
    
    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        try:
            logger.info("Initializing Adaptive Intelligence Controller...")
            
            # Initialize components in parallel where possible
            await asyncio.gather(
                self.model_manager.initialize(),
                self.rag_pipeline.initialize(),
                self.memory_system.initialize(),
                self.reasoning_engine.initialize(),
                self.query_classifier.initialize()
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Main entry point for processing queries.
        
        Args:
            query: The user's query
            context: Optional context from previous conversation
            user_preferences: Optional user preferences and settings
            
        Returns:
            ProcessingResult with response and metadata
        """
        start_time = datetime.now()
        
        try:
            # 1. Analyze query to understand what's needed
            analysis = await self._analyze_query(query, context)
            
            # 2. Build enhanced context from multiple sources
            enhanced_context = await self._build_enhanced_context(
                query, analysis, context, user_preferences
            )
            
            # 3. Select optimal processing strategy
            processing_strategy = await self._select_processing_strategy(
                analysis, enhanced_context
            )
            
            # 4. Execute the processing strategy
            result = await self._execute_processing_strategy(
                query, enhanced_context, processing_strategy, analysis
            )
            
            # 5. Post-process: learn, adapt, and store
            await self._post_process_result(
                query, result, analysis, processing_strategy
            )
            
            # 6. Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time, result.mode_used)
            
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error response
            return ProcessingResult(
                response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                tokens_used=0,
                mode_used=ProcessingMode.NON_THINKING,
                metadata={"error": str(e)}
            )
    
    async def _analyze_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Analyze the query to determine processing requirements."""
        try:
            # Use the query classifier to analyze the query
            classification = await self.query_classifier.classify(query, context)
            
            # Determine if current information is needed
            needs_current = await self._needs_current_information(query, classification)
            
            # Determine if complex reasoning is needed
            needs_reasoning = await self._needs_reasoning(query, classification)
            
            # Determine if tools are needed
            needs_tools = await self._needs_tools(query, classification)
            
            return QueryAnalysis(
                query_type=QueryType(classification.get("type", "simple_fact")),
                complexity=classification.get("complexity", 0.5),
                needs_current_info=needs_current,
                needs_reasoning=needs_reasoning,
                needs_tools=needs_tools,
                confidence=classification.get("confidence", 0.8),
                metadata=classification.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            
            # Return conservative analysis
            return QueryAnalysis(
                query_type=QueryType.SIMPLE_FACT,
                complexity=0.5,
                needs_current_info=False,
                needs_reasoning=False,
                needs_tools=False,
                confidence=0.5
            )
    
    async def _build_enhanced_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build enhanced context from multiple sources."""
        enhanced_context = {
            "query": query,
            "analysis": analysis.dict(),
            "original_context": context or {},
            "user_preferences": user_preferences or {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Gather context from different sources in parallel
            context_tasks = []
            
            # 1. Retrieve relevant memories
            context_tasks.append(
                self.memory_system.retrieve_relevant(query, k=10)
            )
            
            # 2. Get current information if needed
            if analysis.needs_current_info:
                context_tasks.append(
                    self.rag_pipeline.retrieve_current_context(query)
                )
            
            # 3. Get reasoning patterns if needed
            if analysis.needs_reasoning:
                context_tasks.append(
                    self.reasoning_engine.get_relevant_patterns(query)
                )
            
            # Execute all context gathering tasks
            context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
            
            # Process results
            result_index = 0
            
            # Add memory context
            if not isinstance(context_results[result_index], Exception):
                enhanced_context["memories"] = context_results[result_index]
            result_index += 1
            
            # Add current information
            if analysis.needs_current_info:
                if not isinstance(context_results[result_index], Exception):
                    enhanced_context["current_info"] = context_results[result_index]
                result_index += 1
            
            # Add reasoning patterns
            if analysis.needs_reasoning:
                if not isinstance(context_results[result_index], Exception):
                    enhanced_context["reasoning_patterns"] = context_results[result_index]
                result_index += 1
            
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Error building enhanced context: {e}")
            return enhanced_context
    
    async def _select_processing_strategy(
        self,
        analysis: QueryAnalysis,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the optimal processing strategy based on analysis."""
        strategy = {
            "mode": ProcessingMode.NON_THINKING,
            "use_rag": analysis.needs_current_info,
            "use_reasoning": analysis.needs_reasoning,
            "use_tools": analysis.needs_tools,
            "model_params": {},
            "confidence_threshold": 0.7
        }
        
        # Determine processing mode
        if analysis.query_type in [QueryType.COMPLEX_REASONING, QueryType.CODE_GENERATION]:
            strategy["mode"] = ProcessingMode.THINKING
        elif analysis.complexity > self.complexity_threshold:
            strategy["mode"] = ProcessingMode.THINKING
        elif analysis.needs_reasoning:
            strategy["mode"] = ProcessingMode.HYBRID
        
        # Adjust model parameters based on strategy
        if strategy["mode"] == ProcessingMode.THINKING:
            strategy["model_params"] = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4096
            }
        else:
            strategy["model_params"] = {
                "temperature": 0.3,
                "top_p": 0.8,
                "max_tokens": 2048
            }
        
        # Adjust based on user preferences
        user_prefs = context.get("user_preferences", {})
        if user_prefs.get("prefer_detailed_responses"):
            strategy["mode"] = ProcessingMode.THINKING
        
        return strategy
    
    async def _execute_processing_strategy(
        self,
        query: str,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> ProcessingResult:
        """Execute the selected processing strategy."""
        try:
            mode = strategy["mode"]
            
            if mode == ProcessingMode.THINKING:
                return await self._process_with_thinking(
                    query, context, strategy, analysis
                )
            elif mode == ProcessingMode.HYBRID:
                return await self._process_hybrid(
                    query, context, strategy, analysis
                )
            else:
                return await self._process_non_thinking(
                    query, context, strategy, analysis
                )
                
        except Exception as e:
            logger.error(f"Error executing processing strategy: {e}")
            raise
    
    async def _process_with_thinking(
        self,
        query: str,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> ProcessingResult:
        """Process query using thinking mode for complex reasoning."""
        # Build thinking prompt
        thinking_prompt = self._build_thinking_prompt(query, context, analysis)
        
        # Generate response with thinking
        response = await self.model_manager.generate_with_thinking(
            prompt=thinking_prompt,
            **strategy["model_params"]
        )
        
        # Extract thinking process and final answer
        thinking_process = response.get("thinking", "")
        final_answer = response.get("answer", "")
        
        # If reasoning is needed, enhance with reasoning engine
        if strategy["use_reasoning"]:
            reasoning_result = await self.reasoning_engine.enhance_response(
                query, final_answer, thinking_process, context
            )
            final_answer = reasoning_result.get("enhanced_answer", final_answer)
            thinking_process = reasoning_result.get("enhanced_thinking", thinking_process)
        
        return ProcessingResult(
            response=final_answer,
            thinking_process=thinking_process,
            sources=self._extract_sources(context),
            confidence=response.get("confidence", 0.8),
            tokens_used=response.get("tokens_used", 0),
            mode_used=ProcessingMode.THINKING,
            metadata={
                "strategy": strategy,
                "analysis": analysis.dict(),
                "reasoning_used": strategy["use_reasoning"]
            }
        )
    
    async def _process_hybrid(
        self,
        query: str,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> ProcessingResult:
        """Process query using hybrid mode - quick analysis then detailed response."""
        # First, quick analysis to determine if thinking is really needed
        quick_analysis = await self.model_manager.quick_analyze(query, context)
        
        if quick_analysis.get("needs_thinking", False):
            return await self._process_with_thinking(query, context, strategy, analysis)
        else:
            return await self._process_non_thinking(query, context, strategy, analysis)
    
    async def _process_non_thinking(
        self,
        query: str,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> ProcessingResult:
        """Process query using non-thinking mode for quick responses."""
        # Build standard prompt
        prompt = self._build_standard_prompt(query, context, analysis)
        
        # Generate response
        response = await self.model_manager.generate_response(
            prompt=prompt,
            **strategy["model_params"]
        )
        
        return ProcessingResult(
            response=response.get("text", ""),
            sources=self._extract_sources(context),
            confidence=response.get("confidence", 0.7),
            tokens_used=response.get("tokens_used", 0),
            mode_used=ProcessingMode.NON_THINKING,
            metadata={
                "strategy": strategy,
                "analysis": analysis.dict()
            }
        )
    
    async def _post_process_result(
        self,
        query: str,
        result: ProcessingResult,
        analysis: QueryAnalysis,
        strategy: Dict[str, Any]
    ) -> None:
        """Post-process the result: learn, adapt, and store."""
        try:
            # Store the interaction in memory
            await self.memory_system.store_interaction(
                query=query,
                response=result.response,
                thinking_process=result.thinking_process,
                metadata={
                    "analysis": analysis.dict(),
                    "strategy": strategy,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "mode_used": result.mode_used.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Learn from the interaction if it was successful
            if result.confidence > 0.7:
                await self._learn_from_interaction(query, result, analysis, strategy)
            
            # Adapt thresholds based on performance
            await self._adapt_thresholds(result, analysis, strategy)
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
    
    async def _learn_from_interaction(
        self,
        query: str,
        result: ProcessingResult,
        analysis: QueryAnalysis,
        strategy: Dict[str, Any]
    ) -> None:
        """Learn from successful interactions to improve future performance."""
        # Update query classification patterns
        await self.query_classifier.update_patterns(
            query, analysis.query_type, result.confidence
        )
        
        # Update reasoning patterns if reasoning was used
        if strategy["use_reasoning"] and result.thinking_process:
            await self.reasoning_engine.learn_pattern(
                query, result.thinking_process, result.confidence
            )
        
        # Update mode selection patterns
        await self._update_mode_selection_patterns(
            analysis, strategy, result.confidence
        )
    
    async def _adapt_thresholds(
        self,
        result: ProcessingResult,
        analysis: QueryAnalysis,
        strategy: Dict[str, Any]
    ) -> None:
        """Adapt decision thresholds based on performance."""
        # Adjust complexity threshold based on results
        if result.confidence > 0.9 and strategy["mode"] == ProcessingMode.THINKING:
            # If thinking mode worked very well, slightly lower threshold
            self.complexity_threshold = max(
                0.3, 
                self.complexity_threshold - self.adaptive_learning_rate
            )
        elif result.confidence < 0.6 and strategy["mode"] == ProcessingMode.NON_THINKING:
            # If non-thinking mode failed, raise threshold
            self.complexity_threshold = min(
                0.9,
                self.complexity_threshold + self.adaptive_learning_rate
            )
    
    def _build_thinking_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> str:
        """Build a prompt for thinking mode."""
        prompt_parts = [
            "<|thinking|>",
            f"I need to carefully analyze this query: {query}",
            "",
            "Let me consider:",
            "1. What is the user really asking?",
            "2. What information do I need to answer this well?",
            "3. How should I structure my response?",
            "4. What are the potential complications or edge cases?",
            ""
        ]
        
        # Add relevant context
        if context.get("memories"):
            prompt_parts.extend([
                "Relevant memories:",
                json.dumps(context["memories"], indent=2),
                ""
            ])
        
        if context.get("current_info"):
            prompt_parts.extend([
                "Current information:",
                json.dumps(context["current_info"], indent=2),
                ""
            ])
        
        prompt_parts.extend([
            "Let me work through this step by step...",
            "</|thinking|>",
            "",
            f"Query: {query}",
            "",
            "Response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_standard_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> str:
        """Build a standard prompt for non-thinking mode."""
        prompt_parts = [
            "You are an intelligent AI assistant. Answer the user's query clearly and concisely.",
            ""
        ]
        
        # Add relevant context
        if context.get("memories"):
            prompt_parts.extend([
                "Relevant context:",
                json.dumps(context["memories"], indent=2),
                ""
            ])
        
        prompt_parts.extend([
            f"Query: {query}",
            "",
            "Response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_sources(self, context: Dict[str, Any]) -> List[str]:
        """Extract sources from context."""
        sources = []
        
        if context.get("current_info", {}).get("sources"):
            sources.extend(context["current_info"]["sources"])
        
        if context.get("memories", {}).get("sources"):
            sources.extend(context["memories"]["sources"])
        
        return list(set(sources))  # Remove duplicates
    
    def _update_performance_metrics(
        self,
        processing_time: float,
        mode_used: ProcessingMode
    ) -> None:
        """Update performance tracking metrics."""
        self.query_count += 1
        self.total_processing_time += processing_time
        self.mode_usage_stats[mode_used] += 1
    
    async def _needs_current_information(
        self,
        query: str,
        classification: Dict[str, Any]
    ) -> bool:
        """Determine if current information is needed."""
        current_keywords = [
            "today", "latest", "recent", "current", "now", "2024", "2025",
            "news", "update", "what's new", "trending"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in current_keywords)
    
    async def _needs_reasoning(
        self,
        query: str,
        classification: Dict[str, Any]
    ) -> bool:
        """Determine if complex reasoning is needed."""
        reasoning_keywords = [
            "why", "how", "explain", "analyze", "compare", "evaluate",
            "solve", "problem", "debug", "optimize", "design"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in reasoning_keywords)
    
    async def _needs_tools(
        self,
        query: str,
        classification: Dict[str, Any]
    ) -> bool:
        """Determine if tools are needed."""
        tool_keywords = [
            "file", "code", "run", "execute", "search", "web",
            "git", "commit", "push", "pull", "clone"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in tool_keywords)
    
    async def _update_mode_selection_patterns(
        self,
        analysis: QueryAnalysis,
        strategy: Dict[str, Any],
        confidence: float
    ) -> None:
        """Update patterns for mode selection."""
        # This would update internal patterns based on success/failure
        # Implementation would depend on the specific learning mechanism
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = (
            self.total_processing_time / self.query_count 
            if self.query_count > 0 else 0
        )
        
        return {
            "total_queries": self.query_count,
            "average_processing_time": avg_processing_time,
            "mode_usage": {
                mode.value: count for mode, count in self.mode_usage_stats.items()
            },
            "complexity_threshold": self.complexity_threshold
        }
    
    async def shutdown(self) -> None:
        """Shutdown the controller and cleanup resources."""
        try:
            logger.info("Shutting down Adaptive Intelligence Controller...")
            
            # Shutdown components
            await asyncio.gather(
                self.model_manager.shutdown(),
                self.rag_pipeline.shutdown(),
                self.memory_system.shutdown(),
                self.reasoning_engine.shutdown(),
                return_exceptions=True
            )
            
            logger.info("Adaptive Intelligence Controller shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise