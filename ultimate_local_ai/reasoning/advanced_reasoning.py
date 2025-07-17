"""Advanced Reasoning Engine with self-reflection and continuous improvement."""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..models.model_manager import ModelManager
from ..utils.pattern_matcher import PatternMatcher
from ..utils.quality_evaluator import QualityEvaluator

logger = logging.getLogger(__name__)


class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""
    step_number: int
    content: str
    type: str = "reasoning"  # reasoning, observation, conclusion, hypothesis
    confidence: float = Field(ge=0.0, le=1.0)
    dependencies: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningChain(BaseModel):
    """A complete reasoning chain."""
    id: str
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = "chain_of_thought"
    reflection_notes: List[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    timestamp: datetime = Field(default_factory=datetime.now)


class ReflectionResult(BaseModel):
    """Result of self-reflection analysis."""
    needs_revision: bool = False
    quality_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    confidence_adjustment: float = 0.0


class ReasoningPattern(BaseModel):
    """A learned reasoning pattern."""
    id: str
    pattern_type: str
    trigger_conditions: List[str]
    reasoning_template: str
    success_rate: float = Field(ge=0.0, le=1.0)
    usage_count: int = 0
    last_used: datetime = Field(default_factory=datetime.now)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class AdvancedReasoningEngine:
    """
    Advanced reasoning engine with self-reflection capabilities.
    
    This engine provides:
    - Chain-of-thought reasoning
    - Self-reflection and error correction
    - Pattern learning and optimization
    - Multiple reasoning strategies
    - Quality assessment and improvement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_manager = ModelManager(config)
        self.pattern_matcher = PatternMatcher(config)
        self.quality_evaluator = QualityEvaluator(config)
        
        # Reasoning patterns storage
        self.patterns: Dict[str, ReasoningPattern] = {}
        
        # Performance tracking
        self.reasoning_count = 0
        self.successful_reasoning = 0
        self.reflection_improvements = 0
        
        # Settings
        self.max_reasoning_steps = config.get("max_reasoning_steps", 10)
        self.reflection_threshold = config.get("reflection_threshold", 0.7)
        self.quality_threshold = config.get("quality_threshold", 0.8)
        self.pattern_learning_rate = config.get("pattern_learning_rate", 0.1)
        
        logger.info("Advanced Reasoning Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the reasoning engine."""
        try:
            logger.info("Initializing Advanced Reasoning Engine...")
            
            # Initialize components
            await self.model_manager.initialize()
            await self.pattern_matcher.initialize()
            await self.quality_evaluator.initialize()
            
            # Load existing patterns
            await self._load_reasoning_patterns()
            
            logger.info("Advanced Reasoning Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            raise
    
    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        method: str = "auto"
    ) -> ReasoningChain:
        """
        Perform advanced reasoning on a query.
        
        Args:
            query: The query to reason about
            context: Additional context for reasoning
            method: Reasoning method to use
            
        Returns:
            Complete reasoning chain with steps and conclusions
        """
        self.reasoning_count += 1
        
        try:
            # Select reasoning method
            if method == "auto":
                method = await self._select_reasoning_method(query, context)
            
            # Check for applicable patterns
            applicable_pattern = await self._find_applicable_pattern(query, context)
            
            # Perform initial reasoning
            reasoning_chain = await self._perform_initial_reasoning(
                query, context, method, applicable_pattern
            )
            
            # Self-reflection and improvement
            reflection_result = await self._perform_self_reflection(
                reasoning_chain, query, context
            )
            
            # Revise if needed
            if reflection_result.needs_revision:
                logger.info("Revising reasoning based on self-reflection")
                reasoning_chain = await self._revise_reasoning(
                    reasoning_chain, reflection_result, query, context
                )
                self.reflection_improvements += 1
            
            # Update pattern learning
            if reasoning_chain.quality_score > self.quality_threshold:
                await self._learn_from_reasoning(reasoning_chain, query, context)
                self.successful_reasoning += 1
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            
            # Return minimal reasoning chain on error
            return ReasoningChain(
                id=f"error_{datetime.now().timestamp()}",
                query=query,
                steps=[],
                final_answer=f"I encountered an error while reasoning: {str(e)}",
                confidence=0.0,
                method=method,
                quality_score=0.0
            )
    
    async def _select_reasoning_method(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Select the appropriate reasoning method."""
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Mathematical or logical problems
        if any(word in query_lower for word in ["solve", "calculate", "prove", "logical"]):
            return "step_by_step"
        
        # Causal reasoning
        if any(word in query_lower for word in ["why", "because", "cause", "effect"]):
            return "causal"
        
        # Comparative analysis
        if any(word in query_lower for word in ["compare", "contrast", "versus", "better"]):
            return "comparative"
        
        # Creative or open-ended
        if any(word in query_lower for word in ["creative", "brainstorm", "ideas", "imagine"]):
            return "creative"
        
        # Default to chain of thought
        return "chain_of_thought"
    
    async def _find_applicable_pattern(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[ReasoningPattern]:
        """Find applicable reasoning pattern."""
        try:
            # Check patterns for matching trigger conditions
            for pattern in self.patterns.values():
                if await self.pattern_matcher.matches(
                    query, pattern.trigger_conditions, context
                ):
                    # Update usage statistics
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
                    
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding applicable pattern: {e}")
            return None
    
    async def _perform_initial_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        method: str,
        pattern: Optional[ReasoningPattern] = None
    ) -> ReasoningChain:
        """Perform initial reasoning pass."""
        try:
            # Build reasoning prompt
            prompt = await self._build_reasoning_prompt(
                query, context, method, pattern
            )
            
            # Generate reasoning with thinking mode
            response = await self.model_manager.generate_with_thinking(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Parse reasoning steps
            steps = self._parse_reasoning_steps(response.get("thinking", ""))
            
            # Extract final answer
            final_answer = response.get("answer", "")
            
            # Calculate confidence
            confidence = self._calculate_confidence(steps, final_answer)
            
            reasoning_chain = ReasoningChain(
                id=f"reasoning_{datetime.now().timestamp()}",
                query=query,
                steps=steps,
                final_answer=final_answer,
                confidence=confidence,
                method=method
            )
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error in initial reasoning: {e}")
            raise
    
    async def _build_reasoning_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        method: str,
        pattern: Optional[ReasoningPattern] = None
    ) -> str:
        """Build reasoning prompt based on method and pattern."""
        prompt_parts = [
            "<|thinking|>",
            f"I need to reason about this query: {query}",
            "",
            f"Method: {method}",
            ""
        ]
        
        # Add context if available
        if context:
            prompt_parts.extend([
                "Context:",
                json.dumps(context, indent=2),
                ""
            ])
        
        # Add pattern template if available
        if pattern:
            prompt_parts.extend([
                f"Using reasoning pattern: {pattern.pattern_type}",
                "Pattern template:",
                pattern.reasoning_template,
                ""
            ])
        
        # Add method-specific instructions
        if method == "step_by_step":
            prompt_parts.extend([
                "I'll work through this step by step:",
                "1. First, let me understand what's being asked",
                "2. Then, I'll break down the problem into smaller parts",
                "3. Next, I'll solve each part systematically",
                "4. Finally, I'll combine the results into a complete answer",
                ""
            ])
        
        elif method == "causal":
            prompt_parts.extend([
                "I'll analyze the causal relationships:",
                "1. What are the potential causes?",
                "2. What are the effects?",
                "3. What are the mechanisms connecting them?",
                "4. What evidence supports these connections?",
                ""
            ])
        
        elif method == "comparative":
            prompt_parts.extend([
                "I'll make a systematic comparison:",
                "1. What are the key dimensions to compare?",
                "2. How do the options differ on each dimension?",
                "3. What are the trade-offs?",
                "4. What is the overall assessment?",
                ""
            ])
        
        else:  # chain_of_thought
            prompt_parts.extend([
                "Let me think through this carefully:",
                "1. What do I know about this topic?",
                "2. What are the key considerations?",
                "3. How do these factors interact?",
                "4. What conclusion can I draw?",
                ""
            ])
        
        prompt_parts.extend([
            "Let me work through this systematically...",
            "</|thinking|>",
            "",
            f"Query: {query}",
            "",
            "Let me reason through this step by step:",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_reasoning_steps(self, thinking_text: str) -> List[ReasoningStep]:
        """Parse reasoning steps from thinking text."""
        steps = []
        
        try:
            # Split by numbered steps or logical breaks
            step_patterns = [
                r"(\d+)\.\s*([^0-9]+?)(?=\d+\.|$)",  # Numbered steps
                r"(Step \d+:?\s*)(.*?)(?=Step \d+:|$)",  # Step N: format
                r"(First,|Second,|Third,|Next,|Then,|Finally,)\s*(.*?)(?=First,|Second,|Third,|Next,|Then,|Finally,|$)"  # Transition words
            ]
            
            step_number = 1
            
            for pattern in step_patterns:
                matches = re.finditer(pattern, thinking_text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    content = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    content = content.strip()
                    
                    if content and len(content) > 10:  # Only meaningful steps
                        step = ReasoningStep(
                            step_number=step_number,
                            content=content,
                            type="reasoning",
                            confidence=0.8,  # Default confidence
                            dependencies=[],
                            metadata={"source": "thinking"}
                        )
                        
                        steps.append(step)
                        step_number += 1
                
                if steps:  # If we found steps with this pattern, stop
                    break
            
            # If no structured steps found, create from paragraphs
            if not steps:
                paragraphs = [p.strip() for p in thinking_text.split("\n\n") if p.strip()]
                
                for i, paragraph in enumerate(paragraphs[:self.max_reasoning_steps]):
                    if len(paragraph) > 20:  # Only substantial paragraphs
                        step = ReasoningStep(
                            step_number=i + 1,
                            content=paragraph,
                            type="reasoning",
                            confidence=0.7,
                            dependencies=[],
                            metadata={"source": "paragraph"}
                        )
                        steps.append(step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Error parsing reasoning steps: {e}")
            return []
    
    def _calculate_confidence(
        self,
        steps: List[ReasoningStep],
        final_answer: str
    ) -> float:
        """Calculate overall confidence in reasoning."""
        if not steps:
            return 0.3
        
        # Base confidence from step quality
        step_confidence = sum(step.confidence for step in steps) / len(steps)
        
        # Adjust based on reasoning depth
        depth_bonus = min(0.2, len(steps) * 0.05)
        
        # Adjust based on answer quality
        answer_quality = min(0.2, len(final_answer) / 500)  # Longer answers tend to be more detailed
        
        total_confidence = min(1.0, step_confidence + depth_bonus + answer_quality)
        
        return total_confidence
    
    async def _perform_self_reflection(
        self,
        reasoning_chain: ReasoningChain,
        query: str,
        context: Dict[str, Any]
    ) -> ReflectionResult:
        """Perform self-reflection on the reasoning."""
        try:
            # Skip reflection if confidence is already high
            if reasoning_chain.confidence > 0.9:
                return ReflectionResult(
                    needs_revision=False,
                    quality_score=reasoning_chain.confidence,
                    issues=[],
                    insights=[],
                    suggestions=[]
                )
            
            # Build reflection prompt
            reflection_prompt = self._build_reflection_prompt(
                reasoning_chain, query, context
            )
            
            # Generate reflection
            response = await self.model_manager.generate_response(
                prompt=reflection_prompt,
                max_tokens=1024,
                temperature=0.3
            )
            
            reflection_text = response.get("text", "")
            
            # Parse reflection result
            reflection_result = self._parse_reflection_result(reflection_text)
            
            # Evaluate quality
            quality_score = await self.quality_evaluator.evaluate_reasoning(
                reasoning_chain, query, context
            )
            
            reflection_result.quality_score = quality_score
            reflection_result.needs_revision = quality_score < self.reflection_threshold
            
            return reflection_result
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            
            return ReflectionResult(
                needs_revision=False,
                quality_score=reasoning_chain.confidence,
                issues=[f"Reflection error: {str(e)}"],
                insights=[],
                suggestions=[]
            )
    
    def _build_reflection_prompt(
        self,
        reasoning_chain: ReasoningChain,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build reflection prompt."""
        steps_text = "\n".join([
            f"Step {step.step_number}: {step.content}"
            for step in reasoning_chain.steps
        ])
        
        return f"""
Please critically evaluate this reasoning:

Original Query: {query}

Reasoning Steps:
{steps_text}

Final Answer: {reasoning_chain.final_answer}

Evaluate this reasoning and identify:
1. Any logical gaps or errors
2. Missing considerations
3. Potential improvements
4. Overall quality assessment

Provide your evaluation in this format:
ISSUES: [list any problems found]
INSIGHTS: [any valuable observations]
SUGGESTIONS: [specific improvements]
QUALITY: [score from 0-1]
"""
    
    def _parse_reflection_result(self, reflection_text: str) -> ReflectionResult:
        """Parse reflection result from text."""
        try:
            issues = []
            insights = []
            suggestions = []
            quality_score = 0.7  # Default
            
            # Extract issues
            issues_match = re.search(r"ISSUES:\s*(.+?)(?=INSIGHTS:|SUGGESTIONS:|QUALITY:|$)", reflection_text, re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                issues = [issue.strip() for issue in issues_text.split("\n") if issue.strip()]
            
            # Extract insights
            insights_match = re.search(r"INSIGHTS:\s*(.+?)(?=SUGGESTIONS:|QUALITY:|$)", reflection_text, re.DOTALL)
            if insights_match:
                insights_text = insights_match.group(1).strip()
                insights = [insight.strip() for insight in insights_text.split("\n") if insight.strip()]
            
            # Extract suggestions
            suggestions_match = re.search(r"SUGGESTIONS:\s*(.+?)(?=QUALITY:|$)", reflection_text, re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1).strip()
                suggestions = [suggestion.strip() for suggestion in suggestions_text.split("\n") if suggestion.strip()]
            
            # Extract quality score
            quality_match = re.search(r"QUALITY:\s*([0-9.]+)", reflection_text)
            if quality_match:
                quality_score = float(quality_match.group(1))
            
            return ReflectionResult(
                needs_revision=len(issues) > 0 or quality_score < self.reflection_threshold,
                quality_score=quality_score,
                issues=issues,
                insights=insights,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error parsing reflection result: {e}")
            
            return ReflectionResult(
                needs_revision=False,
                quality_score=0.7,
                issues=[],
                insights=[],
                suggestions=[]
            )
    
    async def _revise_reasoning(
        self,
        original_chain: ReasoningChain,
        reflection: ReflectionResult,
        query: str,
        context: Dict[str, Any]
    ) -> ReasoningChain:
        """Revise reasoning based on reflection."""
        try:
            # Build revision prompt
            revision_prompt = self._build_revision_prompt(
                original_chain, reflection, query, context
            )
            
            # Generate revised reasoning
            response = await self.model_manager.generate_with_thinking(
                prompt=revision_prompt,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Parse revised reasoning
            revised_steps = self._parse_reasoning_steps(response.get("thinking", ""))
            revised_answer = response.get("answer", "")
            
            # Calculate new confidence
            revised_confidence = self._calculate_confidence(revised_steps, revised_answer)
            
            # Create revised chain
            revised_chain = ReasoningChain(
                id=f"revised_{original_chain.id}",
                query=query,
                steps=revised_steps,
                final_answer=revised_answer,
                confidence=revised_confidence,
                method=original_chain.method,
                reflection_notes=reflection.suggestions,
                quality_score=reflection.quality_score
            )
            
            return revised_chain
            
        except Exception as e:
            logger.error(f"Error revising reasoning: {e}")
            return original_chain
    
    def _build_revision_prompt(
        self,
        original_chain: ReasoningChain,
        reflection: ReflectionResult,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build revision prompt."""
        issues_text = "\n".join(f"- {issue}" for issue in reflection.issues)
        suggestions_text = "\n".join(f"- {suggestion}" for suggestion in reflection.suggestions)
        
        return f"""
<|thinking|>
I need to revise my previous reasoning about: {query}

Issues identified with my previous reasoning:
{issues_text}

Suggestions for improvement:
{suggestions_text}

Let me reconsider this more carefully, addressing these issues...
</|thinking|>

Let me reconsider this query with the identified improvements:

{query}

Taking into account the feedback, let me reason through this again:
"""
    
    async def _learn_from_reasoning(
        self,
        reasoning_chain: ReasoningChain,
        query: str,
        context: Dict[str, Any]
    ) -> None:
        """Learn from successful reasoning to improve future performance."""
        try:
            # Extract pattern from successful reasoning
            pattern_id = f"pattern_{datetime.now().timestamp()}"
            
            # Identify pattern type
            pattern_type = self._identify_pattern_type(query, reasoning_chain)
            
            # Extract trigger conditions
            trigger_conditions = self._extract_trigger_conditions(query, context)
            
            # Create reasoning template
            reasoning_template = self._create_reasoning_template(reasoning_chain)
            
            # Create new pattern
            new_pattern = ReasoningPattern(
                id=pattern_id,
                pattern_type=pattern_type,
                trigger_conditions=trigger_conditions,
                reasoning_template=reasoning_template,
                success_rate=reasoning_chain.confidence,
                usage_count=1,
                examples=[{
                    "query": query,
                    "context": context,
                    "confidence": reasoning_chain.confidence,
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
            # Store pattern
            self.patterns[pattern_id] = new_pattern
            
            # Update existing similar patterns
            await self._update_similar_patterns(new_pattern, reasoning_chain)
            
            logger.info(f"Learned new reasoning pattern: {pattern_type}")
            
        except Exception as e:
            logger.error(f"Error learning from reasoning: {e}")
    
    def _identify_pattern_type(
        self,
        query: str,
        reasoning_chain: ReasoningChain
    ) -> str:
        """Identify the type of reasoning pattern."""
        query_lower = query.lower()
        
        # Check for specific patterns
        if any(word in query_lower for word in ["solve", "calculate", "math"]):
            return "mathematical"
        elif any(word in query_lower for word in ["debug", "error", "fix"]):
            return "debugging"
        elif any(word in query_lower for word in ["compare", "versus", "better"]):
            return "comparative"
        elif any(word in query_lower for word in ["why", "because", "cause"]):
            return "causal"
        elif any(word in query_lower for word in ["design", "create", "build"]):
            return "creative"
        else:
            return "general"
    
    def _extract_trigger_conditions(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract trigger conditions for pattern matching."""
        conditions = []
        
        # Extract keywords from query
        keywords = query.lower().split()
        important_keywords = [
            word for word in keywords 
            if len(word) > 3 and word not in ["what", "where", "when", "why", "how", "the", "and", "or", "but"]
        ]
        
        conditions.extend(important_keywords[:5])  # Top 5 keywords
        
        # Add context-based conditions
        if context.get("domain"):
            conditions.append(f"domain:{context['domain']}")
        
        if context.get("complexity"):
            conditions.append(f"complexity:{context['complexity']}")
        
        return conditions
    
    def _create_reasoning_template(
        self,
        reasoning_chain: ReasoningChain
    ) -> str:
        """Create reasoning template from successful chain."""
        template_parts = []
        
        for step in reasoning_chain.steps:
            # Generalize step content
            generalized_content = self._generalize_step_content(step.content)
            template_parts.append(f"Step {step.step_number}: {generalized_content}")
        
        return "\n".join(template_parts)
    
    def _generalize_step_content(self, content: str) -> str:
        """Generalize step content to create reusable template."""
        # Replace specific values with placeholders
        generalized = re.sub(r'\d+', '[NUMBER]', content)
        generalized = re.sub(r'"[^"]*"', '[QUOTED_TEXT]', generalized)
        generalized = re.sub(r'\b[A-Z][a-z]+\b', '[PROPER_NOUN]', generalized)
        
        return generalized
    
    async def _update_similar_patterns(
        self,
        new_pattern: ReasoningPattern,
        reasoning_chain: ReasoningChain
    ) -> None:
        """Update similar existing patterns."""
        try:
            for pattern in self.patterns.values():
                if pattern.pattern_type == new_pattern.pattern_type:
                    # Update success rate with exponential moving average
                    pattern.success_rate = (
                        pattern.success_rate * 0.8 + 
                        reasoning_chain.confidence * 0.2
                    )
                    
                    # Add example
                    if len(pattern.examples) < 10:  # Keep only recent examples
                        pattern.examples.append(new_pattern.examples[0])
                    else:
                        pattern.examples.pop(0)
                        pattern.examples.append(new_pattern.examples[0])
                    
                    break
                    
        except Exception as e:
            logger.error(f"Error updating similar patterns: {e}")
    
    async def _load_reasoning_patterns(self) -> None:
        """Load existing reasoning patterns."""
        try:
            # This would load patterns from persistent storage
            # For now, initialize with basic patterns
            
            basic_patterns = {
                "mathematical": ReasoningPattern(
                    id="math_basic",
                    pattern_type="mathematical",
                    trigger_conditions=["solve", "calculate", "math", "number"],
                    reasoning_template="1. Identify the problem type\n2. Break down into steps\n3. Apply relevant formulas\n4. Calculate result\n5. Verify answer",
                    success_rate=0.8,
                    usage_count=0
                ),
                "debugging": ReasoningPattern(
                    id="debug_basic",
                    pattern_type="debugging",
                    trigger_conditions=["debug", "error", "fix", "problem"],
                    reasoning_template="1. Identify the error\n2. Analyze the context\n3. Consider possible causes\n4. Test hypotheses\n5. Implement solution",
                    success_rate=0.75,
                    usage_count=0
                )
            }
            
            self.patterns.update(basic_patterns)
            
            logger.info(f"Loaded {len(self.patterns)} reasoning patterns")
            
        except Exception as e:
            logger.error(f"Error loading reasoning patterns: {e}")
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine if reasoning is needed."""
        try:
            # Simple analysis for now
            complexity_indicators = [
                "why", "how", "explain", "analyze", "compare", "solve",
                "debug", "optimize", "design", "evaluate", "prove"
            ]
            
            query_lower = query.lower()
            needs_reasoning = any(indicator in query_lower for indicator in complexity_indicators)
            
            complexity = len([ind for ind in complexity_indicators if ind in query_lower]) / len(complexity_indicators)
            
            return {
                "needs_reasoning": needs_reasoning,
                "complexity": complexity,
                "method": await self._select_reasoning_method(query, {}),
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {"needs_reasoning": False, "complexity": 0.5, "method": "chain_of_thought"}
    
    async def get_relevant_patterns(self, query: str) -> List[ReasoningPattern]:
        """Get relevant patterns for a query."""
        try:
            relevant_patterns = []
            
            for pattern in self.patterns.values():
                if await self.pattern_matcher.matches(query, pattern.trigger_conditions, {}):
                    relevant_patterns.append(pattern)
            
            # Sort by success rate
            relevant_patterns.sort(key=lambda p: p.success_rate, reverse=True)
            
            return relevant_patterns[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Error getting relevant patterns: {e}")
            return []
    
    async def enhance_response(
        self,
        query: str,
        response: str,
        thinking_process: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance response with reasoning capabilities."""
        try:
            # Analyze if enhancement is needed
            needs_enhancement = len(thinking_process) < 200 or "because" not in response.lower()
            
            if not needs_enhancement:
                return {
                    "enhanced_answer": response,
                    "enhanced_thinking": thinking_process,
                    "enhancement_applied": False
                }
            
            # Apply reasoning enhancement
            enhanced_prompt = f"""
Please enhance this response with better reasoning:

Original Query: {query}
Original Response: {response}

Provide a more detailed explanation with clear reasoning steps.
"""
            
            enhanced_response = await self.model_manager.generate_with_thinking(
                prompt=enhanced_prompt,
                max_tokens=2048,
                temperature=0.7
            )
            
            return {
                "enhanced_answer": enhanced_response.get("answer", response),
                "enhanced_thinking": enhanced_response.get("thinking", thinking_process),
                "enhancement_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return {
                "enhanced_answer": response,
                "enhanced_thinking": thinking_process,
                "enhancement_applied": False
            }
    
    async def learn_pattern(
        self,
        query: str,
        thinking_process: str,
        confidence: float
    ) -> None:
        """Learn a new reasoning pattern."""
        try:
            if confidence > self.quality_threshold:
                # Create mock reasoning chain for learning
                steps = self._parse_reasoning_steps(thinking_process)
                
                reasoning_chain = ReasoningChain(
                    id=f"learn_{datetime.now().timestamp()}",
                    query=query,
                    steps=steps,
                    final_answer="",
                    confidence=confidence,
                    method="learned"
                )
                
                await self._learn_from_reasoning(reasoning_chain, query, {})
                
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_reasoning_count": self.reasoning_count,
            "successful_reasoning": self.successful_reasoning,
            "success_rate": self.successful_reasoning / self.reasoning_count if self.reasoning_count > 0 else 0,
            "reflection_improvements": self.reflection_improvements,
            "improvement_rate": self.reflection_improvements / self.reasoning_count if self.reasoning_count > 0 else 0,
            "patterns_learned": len(self.patterns),
            "quality_threshold": self.quality_threshold
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Save patterns before shutdown
            await self._save_reasoning_patterns()
            
            logger.info("Advanced Reasoning Engine cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _save_reasoning_patterns(self) -> None:
        """Save reasoning patterns to persistent storage."""
        try:
            # This would save patterns to database or file
            # For now, just log the count
            logger.info(f"Saving {len(self.patterns)} reasoning patterns")
            
        except Exception as e:
            logger.error(f"Error saving reasoning patterns: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the reasoning engine."""
        try:
            logger.info("Shutting down Advanced Reasoning Engine...")
            
            await self.cleanup()
            
            logger.info("Advanced Reasoning Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise