"""Integrated Memory System with 4-tier architecture."""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
from pydantic import BaseModel, Field

from ..utils.embeddings import EmbeddingManager
from ..utils.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class MemoryItem(BaseModel):
    """Base class for memory items."""
    id: str
    content: str
    timestamp: datetime
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class WorkingMemoryItem(MemoryItem):
    """Working memory item for current conversation."""
    role: str  # user, assistant, system
    conversation_id: str
    turn_number: int
    context_window_position: int = 0


class EpisodicMemoryItem(MemoryItem):
    """Episodic memory item for recent interactions."""
    conversation_id: str
    query: str
    response: str
    confidence: float = Field(ge=0.0, le=1.0)
    success: bool = True
    emotion: Optional[str] = None
    decay_factor: float = 1.0


class SemanticMemoryItem(MemoryItem):
    """Semantic memory item for facts and knowledge."""
    fact_type: str
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    verification_status: str = "unverified"  # verified, unverified, disputed


class ProceduralMemoryItem(MemoryItem):
    """Procedural memory item for learned patterns."""
    pattern_type: str
    trigger_conditions: List[str]
    action_sequence: List[str]
    success_rate: float = Field(ge=0.0, le=1.0)
    usage_count: int = 0
    context_constraints: Dict[str, Any] = Field(default_factory=dict)


class MemoryConsolidationResult(BaseModel):
    """Result of memory consolidation process."""
    items_processed: int
    items_consolidated: int
    items_forgotten: int
    new_patterns_learned: int
    facts_verified: int
    quality_score: float = Field(ge=0.0, le=1.0)


class IntegratedMemorySystem:
    """
    Multi-tier memory system with human-like memory organization.
    
    Tiers:
    1. Working Memory - Current conversation context (fast access, limited capacity)
    2. Episodic Memory - Recent interactions and experiences (time-decay, emotional weighting)
    3. Semantic Memory - Facts and knowledge (persistent, cross-referenced)
    4. Procedural Memory - Learned patterns and procedures (reinforcement learning)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(config)
        self.similarity_calculator = SimilarityCalculator(config)
        
        # Database connections
        self.db_path = config.get("memory_db_path", "./memory.db")
        self.duck_db_path = config.get("analytics_db_path", "./analytics.duckdb")
        
        # Memory tiers
        self.working_memory: List[WorkingMemoryItem] = []
        self.episodic_memory: List[EpisodicMemoryItem] = []
        self.semantic_memory: List[SemanticMemoryItem] = []
        self.procedural_memory: List[ProceduralMemoryItem] = []
        
        # Settings
        self.working_memory_capacity = config.get("working_memory_capacity", 20)
        self.episodic_memory_capacity = config.get("episodic_memory_capacity", 1000)
        self.semantic_memory_capacity = config.get("semantic_memory_capacity", 10000)
        self.procedural_memory_capacity = config.get("procedural_memory_capacity", 500)
        
        # Decay and consolidation settings
        self.episodic_decay_rate = config.get("episodic_decay_rate", 0.1)
        self.consolidation_threshold = config.get("consolidation_threshold", 0.7)
        self.forgetting_threshold = config.get("forgetting_threshold", 0.1)
        
        # Performance tracking
        self.access_count = 0
        self.consolidation_count = 0
        self.last_consolidation = datetime.now()
        
        logger.info("Integrated Memory System initialized")
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        try:
            logger.info("Initializing Integrated Memory System...")
            
            # Initialize components
            await self.embedding_manager.initialize()
            await self.similarity_calculator.initialize()
            
            # Setup databases
            await self._setup_databases()
            
            # Load existing memories
            await self._load_memories()
            
            logger.info("Integrated Memory System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise
    
    async def _setup_databases(self) -> None:
        """Setup SQLite and DuckDB databases."""
        try:
            # SQLite for structured memory storage
            self.sqlite_conn = sqlite3.connect(self.db_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            
            # DuckDB for analytics
            self.duck_conn = duckdb.connect(self.duck_db_path)
            
            # Create analytics views
            await self._create_analytics_views()
            
        except Exception as e:
            logger.error(f"Error setting up databases: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Working memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            # Episodic memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    success BOOLEAN DEFAULT TRUE,
                    emotion TEXT,
                    decay_factor REAL DEFAULT 1.0,
                    timestamp TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            # Semantic memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id TEXT PRIMARY KEY,
                    fact_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    source TEXT NOT NULL,
                    verification_status TEXT DEFAULT 'unverified',
                    timestamp TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            # Procedural memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memory (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    trigger_conditions TEXT NOT NULL,
                    action_sequence TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    context_constraints TEXT,
                    timestamp TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_working_conversation ON working_memory(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_conversation ON episodic_memory(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_memory(subject)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedural_type ON procedural_memory(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON working_memory(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON episodic_memory(importance)")
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def _create_analytics_views(self) -> None:
        """Create analytics views in DuckDB."""
        try:
            # This would create analytics views for memory usage patterns
            # For now, just create a basic structure
            self.duck_conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_analytics (
                    timestamp TIMESTAMP,
                    memory_type VARCHAR,
                    operation VARCHAR,
                    item_count INTEGER,
                    avg_importance REAL,
                    access_frequency REAL
                )
            """)
            
        except Exception as e:
            logger.error(f"Error creating analytics views: {e}")
            raise
    
    async def _load_memories(self) -> None:
        """Load existing memories from database."""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Load working memory
            cursor.execute("SELECT * FROM working_memory ORDER BY timestamp DESC LIMIT ?", 
                         (self.working_memory_capacity,))
            
            for row in cursor.fetchall():
                item = WorkingMemoryItem(
                    id=row["id"],
                    content=row["content"],
                    role=row["role"],
                    conversation_id=row["conversation_id"],
                    turn_number=row["turn_number"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    importance=row["importance"],
                    access_count=row["access_count"],
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    embedding=json.loads(row["embedding"]) if row["embedding"] else None
                )
                self.working_memory.append(item)
            
            # Load episodic memory
            cursor.execute("SELECT * FROM episodic_memory ORDER BY timestamp DESC LIMIT ?", 
                         (self.episodic_memory_capacity,))
            
            for row in cursor.fetchall():
                item = EpisodicMemoryItem(
                    id=row["id"],
                    content=f"{row['query']} -> {row['response']}",
                    conversation_id=row["conversation_id"],
                    query=row["query"],
                    response=row["response"],
                    confidence=row["confidence"],
                    success=bool(row["success"]),
                    emotion=row["emotion"],
                    decay_factor=row["decay_factor"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    importance=row["importance"],
                    access_count=row["access_count"],
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    embedding=json.loads(row["embedding"]) if row["embedding"] else None
                )
                self.episodic_memory.append(item)
            
            # Load semantic memory (sample)
            cursor.execute("SELECT * FROM semantic_memory ORDER BY importance DESC LIMIT 100")
            
            for row in cursor.fetchall():
                item = SemanticMemoryItem(
                    id=row["id"],
                    content=f"{row['subject']} {row['predicate']} {row['object']}",
                    fact_type=row["fact_type"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    confidence=row["confidence"],
                    source=row["source"],
                    verification_status=row["verification_status"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    importance=row["importance"],
                    access_count=row["access_count"],
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    embedding=json.loads(row["embedding"]) if row["embedding"] else None
                )
                self.semantic_memory.append(item)
            
            # Load procedural memory
            cursor.execute("SELECT * FROM procedural_memory ORDER BY success_rate DESC LIMIT 50")
            
            for row in cursor.fetchall():
                item = ProceduralMemoryItem(
                    id=row["id"],
                    content=f"Pattern: {row['pattern_type']}",
                    pattern_type=row["pattern_type"],
                    trigger_conditions=json.loads(row["trigger_conditions"]),
                    action_sequence=json.loads(row["action_sequence"]),
                    success_rate=row["success_rate"],
                    usage_count=row["usage_count"],
                    context_constraints=json.loads(row["context_constraints"]) if row["context_constraints"] else {},
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    importance=row["importance"],
                    access_count=row["access_count"],
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    embedding=json.loads(row["embedding"]) if row["embedding"] else None
                )
                self.procedural_memory.append(item)
            
            logger.info(f"Loaded memories: Working={len(self.working_memory)}, "
                       f"Episodic={len(self.episodic_memory)}, "
                       f"Semantic={len(self.semantic_memory)}, "
                       f"Procedural={len(self.procedural_memory)}")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            # Continue with empty memories if loading fails
    
    async def store_interaction(
        self,
        query: str,
        response: str,
        thinking_process: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a complete interaction across memory tiers."""
        try:
            interaction_id = f"interaction_{datetime.now().timestamp()}"
            conversation_id = metadata.get("conversation_id", "default") if metadata else "default"
            
            # 1. Add to working memory
            await self._add_to_working_memory(
                content=f"User: {query}\nAssistant: {response}",
                role="interaction",
                conversation_id=conversation_id,
                turn_number=len(self.working_memory) + 1,
                metadata=metadata or {}
            )
            
            # 2. Add to episodic memory
            await self._add_to_episodic_memory(
                conversation_id=conversation_id,
                query=query,
                response=response,
                confidence=metadata.get("confidence", 0.7) if metadata else 0.7,
                success=metadata.get("success", True) if metadata else True,
                metadata=metadata or {}
            )
            
            # 3. Extract facts for semantic memory
            facts = await self._extract_facts(query, response, thinking_process)
            for fact in facts:
                await self._add_to_semantic_memory(**fact)
            
            # 4. Learn patterns for procedural memory
            patterns = await self._extract_patterns(query, response, thinking_process, metadata)
            for pattern in patterns:
                await self._add_to_procedural_memory(**pattern)
            
            # 5. Trigger consolidation if needed
            await self._trigger_consolidation_if_needed()
            
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            raise
    
    async def _add_to_working_memory(
        self,
        content: str,
        role: str,
        conversation_id: str,
        turn_number: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Add item to working memory."""
        try:
            # Generate embedding
            embedding = await self.embedding_manager.get_embedding(content)
            
            # Create memory item
            item = WorkingMemoryItem(
                id=f"working_{datetime.now().timestamp()}",
                content=content,
                role=role,
                conversation_id=conversation_id,
                turn_number=turn_number,
                timestamp=datetime.now(),
                importance=self._calculate_importance(content, metadata),
                metadata=metadata,
                embedding=embedding
            )
            
            # Add to memory
            self.working_memory.append(item)
            
            # Maintain capacity
            if len(self.working_memory) > self.working_memory_capacity:
                # Remove oldest items
                self.working_memory = self.working_memory[-self.working_memory_capacity:]
            
            # Save to database
            await self._save_working_memory_item(item)
            
        except Exception as e:
            logger.error(f"Error adding to working memory: {e}")
            raise
    
    async def _add_to_episodic_memory(
        self,
        conversation_id: str,
        query: str,
        response: str,
        confidence: float,
        success: bool,
        metadata: Dict[str, Any]
    ) -> None:
        """Add item to episodic memory."""
        try:
            # Generate embedding
            content = f"{query} -> {response}"
            embedding = await self.embedding_manager.get_embedding(content)
            
            # Calculate importance
            importance = self._calculate_episodic_importance(
                query, response, confidence, success, metadata
            )
            
            # Create memory item
            item = EpisodicMemoryItem(
                id=f"episodic_{datetime.now().timestamp()}",
                content=content,
                conversation_id=conversation_id,
                query=query,
                response=response,
                confidence=confidence,
                success=success,
                emotion=metadata.get("emotion"),
                timestamp=datetime.now(),
                importance=importance,
                metadata=metadata,
                embedding=embedding
            )
            
            # Add to memory
            self.episodic_memory.append(item)
            
            # Maintain capacity and apply decay
            await self._maintain_episodic_memory()
            
            # Save to database
            await self._save_episodic_memory_item(item)
            
        except Exception as e:
            logger.error(f"Error adding to episodic memory: {e}")
            raise
    
    async def _add_to_semantic_memory(
        self,
        fact_type: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        source: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Add item to semantic memory."""
        try:
            # Generate embedding
            content = f"{subject} {predicate} {object}"
            embedding = await self.embedding_manager.get_embedding(content)
            
            # Check for existing similar facts
            existing_fact = await self._find_similar_semantic_fact(
                subject, predicate, object, confidence
            )
            
            if existing_fact:
                # Update existing fact
                existing_fact.confidence = max(existing_fact.confidence, confidence)
                existing_fact.access_count += 1
                existing_fact.last_accessed = datetime.now()
                await self._save_semantic_memory_item(existing_fact)
            else:
                # Create new fact
                item = SemanticMemoryItem(
                    id=f"semantic_{datetime.now().timestamp()}",
                    content=content,
                    fact_type=fact_type,
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    confidence=confidence,
                    source=source,
                    timestamp=datetime.now(),
                    importance=confidence,
                    metadata=metadata,
                    embedding=embedding
                )
                
                self.semantic_memory.append(item)
                await self._save_semantic_memory_item(item)
            
            # Maintain capacity
            if len(self.semantic_memory) > self.semantic_memory_capacity:
                # Remove lowest confidence facts
                self.semantic_memory.sort(key=lambda x: x.confidence, reverse=True)
                self.semantic_memory = self.semantic_memory[:self.semantic_memory_capacity]
            
        except Exception as e:
            logger.error(f"Error adding to semantic memory: {e}")
            raise
    
    async def _add_to_procedural_memory(
        self,
        pattern_type: str,
        trigger_conditions: List[str],
        action_sequence: List[str],
        success_rate: float,
        context_constraints: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """Add item to procedural memory."""
        try:
            # Generate embedding
            content = f"Pattern: {pattern_type} - {' '.join(trigger_conditions)}"
            embedding = await self.embedding_manager.get_embedding(content)
            
            # Check for existing similar patterns
            existing_pattern = await self._find_similar_procedural_pattern(
                pattern_type, trigger_conditions
            )
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.success_rate = (
                    existing_pattern.success_rate * 0.8 + success_rate * 0.2
                )
                existing_pattern.usage_count += 1
                existing_pattern.last_accessed = datetime.now()
                await self._save_procedural_memory_item(existing_pattern)
            else:
                # Create new pattern
                item = ProceduralMemoryItem(
                    id=f"procedural_{datetime.now().timestamp()}",
                    content=content,
                    pattern_type=pattern_type,
                    trigger_conditions=trigger_conditions,
                    action_sequence=action_sequence,
                    success_rate=success_rate,
                    context_constraints=context_constraints,
                    timestamp=datetime.now(),
                    importance=success_rate,
                    metadata=metadata,
                    embedding=embedding
                )
                
                self.procedural_memory.append(item)
                await self._save_procedural_memory_item(item)
            
            # Maintain capacity
            if len(self.procedural_memory) > self.procedural_memory_capacity:
                # Remove lowest success rate patterns
                self.procedural_memory.sort(key=lambda x: x.success_rate, reverse=True)
                self.procedural_memory = self.procedural_memory[:self.procedural_memory_capacity]
            
        except Exception as e:
            logger.error(f"Error adding to procedural memory: {e}")
            raise
    
    async def retrieve_relevant(
        self,
        query: str,
        k: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant memories from all tiers."""
        try:
            self.access_count += 1
            
            # Generate query embedding
            query_embedding = await self.embedding_manager.get_embedding(query)
            
            # Retrieve from each memory type
            memories = {}
            
            memory_types = memory_types or ["working", "episodic", "semantic", "procedural"]
            
            if "working" in memory_types:
                memories["working"] = await self._retrieve_from_working_memory(
                    query, query_embedding, k // 4
                )
            
            if "episodic" in memory_types:
                memories["episodic"] = await self._retrieve_from_episodic_memory(
                    query, query_embedding, k // 4
                )
            
            if "semantic" in memory_types:
                memories["semantic"] = await self._retrieve_from_semantic_memory(
                    query, query_embedding, k // 4
                )
            
            if "procedural" in memory_types:
                memories["procedural"] = await self._retrieve_from_procedural_memory(
                    query, query_embedding, k // 4
                )
            
            # Consolidate and rank results
            consolidated_memories = await self._consolidate_memories(memories, query)
            
            return consolidated_memories
            
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return {"working": [], "episodic": [], "semantic": [], "procedural": []}
    
    async def _retrieve_from_working_memory(
        self,
        query: str,
        query_embedding: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from working memory."""
        try:
            results = []
            
            for item in self.working_memory:
                if item.embedding:
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, item.embedding
                    )
                    
                    results.append({
                        "id": item.id,
                        "content": item.content,
                        "similarity": similarity,
                        "importance": item.importance,
                        "timestamp": item.timestamp.isoformat(),
                        "metadata": item.metadata,
                        "type": "working"
                    })
            
            # Sort by similarity and importance
            results.sort(key=lambda x: x["similarity"] * 0.7 + x["importance"] * 0.3, reverse=True)
            
            # Update access counts
            for result in results[:k]:
                item = next((i for i in self.working_memory if i.id == result["id"]), None)
                if item:
                    item.access_count += 1
                    item.last_accessed = datetime.now()
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving from working memory: {e}")
            return []
    
    async def _retrieve_from_episodic_memory(
        self,
        query: str,
        query_embedding: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from episodic memory."""
        try:
            results = []
            
            for item in self.episodic_memory:
                if item.embedding:
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, item.embedding
                    )
                    
                    # Apply decay factor
                    age_days = (datetime.now() - item.timestamp).days
                    decay = max(0.1, 1.0 - (age_days * self.episodic_decay_rate))
                    
                    adjusted_similarity = similarity * decay * item.importance
                    
                    results.append({
                        "id": item.id,
                        "content": item.content,
                        "query": item.query,
                        "response": item.response,
                        "similarity": adjusted_similarity,
                        "confidence": item.confidence,
                        "success": item.success,
                        "timestamp": item.timestamp.isoformat(),
                        "metadata": item.metadata,
                        "type": "episodic"
                    })
            
            # Sort by adjusted similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update access counts
            for result in results[:k]:
                item = next((i for i in self.episodic_memory if i.id == result["id"]), None)
                if item:
                    item.access_count += 1
                    item.last_accessed = datetime.now()
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving from episodic memory: {e}")
            return []
    
    async def _retrieve_from_semantic_memory(
        self,
        query: str,
        query_embedding: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from semantic memory."""
        try:
            results = []
            
            for item in self.semantic_memory:
                if item.embedding:
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, item.embedding
                    )
                    
                    # Weight by confidence and verification status
                    verification_weight = {"verified": 1.0, "unverified": 0.8, "disputed": 0.3}
                    weight = verification_weight.get(item.verification_status, 0.8)
                    
                    adjusted_similarity = similarity * item.confidence * weight
                    
                    results.append({
                        "id": item.id,
                        "content": item.content,
                        "subject": item.subject,
                        "predicate": item.predicate,
                        "object": item.object,
                        "similarity": adjusted_similarity,
                        "confidence": item.confidence,
                        "source": item.source,
                        "verification_status": item.verification_status,
                        "timestamp": item.timestamp.isoformat(),
                        "metadata": item.metadata,
                        "type": "semantic"
                    })
            
            # Sort by adjusted similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update access counts
            for result in results[:k]:
                item = next((i for i in self.semantic_memory if i.id == result["id"]), None)
                if item:
                    item.access_count += 1
                    item.last_accessed = datetime.now()
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving from semantic memory: {e}")
            return []
    
    async def _retrieve_from_procedural_memory(
        self,
        query: str,
        query_embedding: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from procedural memory."""
        try:
            results = []
            
            for item in self.procedural_memory:
                if item.embedding:
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, item.embedding
                    )
                    
                    # Weight by success rate and usage
                    usage_weight = min(1.0, item.usage_count / 10)
                    adjusted_similarity = similarity * item.success_rate * (0.7 + usage_weight * 0.3)
                    
                    results.append({
                        "id": item.id,
                        "content": item.content,
                        "pattern_type": item.pattern_type,
                        "trigger_conditions": item.trigger_conditions,
                        "action_sequence": item.action_sequence,
                        "similarity": adjusted_similarity,
                        "success_rate": item.success_rate,
                        "usage_count": item.usage_count,
                        "timestamp": item.timestamp.isoformat(),
                        "metadata": item.metadata,
                        "type": "procedural"
                    })
            
            # Sort by adjusted similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update access counts
            for result in results[:k]:
                item = next((i for i in self.procedural_memory if i.id == result["id"]), None)
                if item:
                    item.access_count += 1
                    item.usage_count += 1
                    item.last_accessed = datetime.now()
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving from procedural memory: {e}")
            return []
    
    async def _consolidate_memories(
        self,
        memories: Dict[str, List[Dict[str, Any]]],
        query: str
    ) -> Dict[str, Any]:
        """Consolidate memories from different tiers."""
        try:
            # Combine all memories
            all_memories = []
            
            for memory_type, items in memories.items():
                for item in items:
                    item["memory_tier"] = memory_type
                    all_memories.append(item)
            
            # Sort by overall relevance
            all_memories.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Extract sources
            sources = list(set([
                item.get("source", "unknown") 
                for item in all_memories 
                if "source" in item
            ]))
            
            return {
                "memories": memories,
                "combined": all_memories,
                "sources": sources,
                "total_items": len(all_memories),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {"memories": memories, "combined": [], "sources": [], "total_items": 0}
    
    def _calculate_importance(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for memory item."""
        importance = 0.5  # Base importance
        
        # Adjust based on content length
        importance += min(0.2, len(content) / 1000)
        
        # Adjust based on metadata
        if metadata.get("user_feedback") == "positive":
            importance += 0.3
        elif metadata.get("user_feedback") == "negative":
            importance -= 0.2
        
        if metadata.get("confidence", 0) > 0.8:
            importance += 0.2
        
        # Ensure bounds
        return max(0.0, min(1.0, importance))
    
    def _calculate_episodic_importance(
        self,
        query: str,
        response: str,
        confidence: float,
        success: bool,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate importance for episodic memory."""
        importance = confidence
        
        # Adjust based on success
        if success:
            importance += 0.2
        else:
            importance -= 0.1
        
        # Adjust based on query complexity
        if len(query.split()) > 10:
            importance += 0.1
        
        # Adjust based on response quality
        if len(response) > 500:
            importance += 0.1
        
        # Emotional importance
        if metadata.get("emotion") in ["joy", "surprise"]:
            importance += 0.2
        elif metadata.get("emotion") in ["anger", "frustration"]:
            importance += 0.1  # Negative emotions are also important to remember
        
        return max(0.0, min(1.0, importance))
    
    async def _extract_facts(
        self,
        query: str,
        response: str,
        thinking_process: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract facts for semantic memory."""
        try:
            # Simple fact extraction - would be more sophisticated in practice
            facts = []
            
            # Look for statements in response
            sentences = response.split('. ')
            
            for sentence in sentences:
                if len(sentence) > 20 and any(verb in sentence.lower() for verb in ['is', 'are', 'was', 'were', 'has', 'have']):
                    # Simple subject-predicate-object extraction
                    words = sentence.split()
                    if len(words) >= 3:
                        subject = words[0]
                        predicate = "is"  # Simplified
                        object_part = " ".join(words[2:])
                        
                        facts.append({
                            "fact_type": "statement",
                            "subject": subject,
                            "predicate": predicate,
                            "object": object_part,
                            "confidence": 0.7,
                            "source": "conversation",
                            "metadata": {"query": query, "response": response}
                        })
            
            return facts[:5]  # Limit to top 5 facts
            
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []
    
    async def _extract_patterns(
        self,
        query: str,
        response: str,
        thinking_process: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract patterns for procedural memory."""
        try:
            patterns = []
            
            # Look for successful problem-solving patterns
            if metadata and metadata.get("success", False) and metadata.get("confidence", 0) > 0.7:
                # Extract query pattern
                query_words = query.lower().split()
                trigger_words = [word for word in query_words if len(word) > 3][:5]
                
                # Extract response pattern
                response_sentences = response.split('. ')[:3]  # First 3 sentences
                
                patterns.append({
                    "pattern_type": "successful_response",
                    "trigger_conditions": trigger_words,
                    "action_sequence": response_sentences,
                    "success_rate": metadata.get("confidence", 0.7),
                    "context_constraints": {
                        "query_length": len(query),
                        "response_length": len(response)
                    },
                    "metadata": metadata or {}
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return []
    
    async def _maintain_episodic_memory(self) -> None:
        """Maintain episodic memory with decay and capacity limits."""
        try:
            # Apply decay
            current_time = datetime.now()
            
            for item in self.episodic_memory:
                age_days = (current_time - item.timestamp).days
                item.decay_factor = max(0.1, 1.0 - (age_days * self.episodic_decay_rate))
                item.importance *= item.decay_factor
            
            # Remove items below forgetting threshold
            self.episodic_memory = [
                item for item in self.episodic_memory 
                if item.importance > self.forgetting_threshold
            ]
            
            # Maintain capacity
            if len(self.episodic_memory) > self.episodic_memory_capacity:
                self.episodic_memory.sort(key=lambda x: x.importance, reverse=True)
                self.episodic_memory = self.episodic_memory[:self.episodic_memory_capacity]
            
        except Exception as e:
            logger.error(f"Error maintaining episodic memory: {e}")
    
    async def _trigger_consolidation_if_needed(self) -> None:
        """Trigger memory consolidation if needed."""
        try:
            # Check if consolidation is needed
            hours_since_last = (datetime.now() - self.last_consolidation).total_seconds() / 3600
            
            if hours_since_last > 24:  # Daily consolidation
                await self._consolidate_memories_background()
                self.last_consolidation = datetime.now()
                self.consolidation_count += 1
            
        except Exception as e:
            logger.error(f"Error triggering consolidation: {e}")
    
    async def _consolidate_memories_background(self) -> MemoryConsolidationResult:
        """Background memory consolidation process."""
        try:
            logger.info("Starting memory consolidation...")
            
            items_processed = 0
            items_consolidated = 0
            items_forgotten = 0
            new_patterns_learned = 0
            facts_verified = 0
            
            # Consolidate episodic to semantic
            for item in self.episodic_memory:
                if item.importance > self.consolidation_threshold:
                    # Extract facts from successful interactions
                    facts = await self._extract_facts(item.query, item.response)
                    
                    for fact in facts:
                        await self._add_to_semantic_memory(**fact)
                        facts_verified += 1
                    
                    items_consolidated += 1
                
                items_processed += 1
            
            # Learn new patterns from successful interactions
            successful_interactions = [
                item for item in self.episodic_memory 
                if item.success and item.confidence > 0.8
            ]
            
            for item in successful_interactions:
                patterns = await self._extract_patterns(
                    item.query, item.response, None, item.metadata
                )
                
                for pattern in patterns:
                    await self._add_to_procedural_memory(**pattern)
                    new_patterns_learned += 1
            
            # Cleanup old memories
            await self._maintain_episodic_memory()
            
            # Calculate quality score
            quality_score = (items_consolidated + new_patterns_learned) / max(1, items_processed)
            
            result = MemoryConsolidationResult(
                items_processed=items_processed,
                items_consolidated=items_consolidated,
                items_forgotten=items_forgotten,
                new_patterns_learned=new_patterns_learned,
                facts_verified=facts_verified,
                quality_score=quality_score
            )
            
            logger.info(f"Memory consolidation complete: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
            return MemoryConsolidationResult(
                items_processed=0,
                items_consolidated=0,
                items_forgotten=0,
                new_patterns_learned=0,
                facts_verified=0,
                quality_score=0.0
            )
    
    async def _find_similar_semantic_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float
    ) -> Optional[SemanticMemoryItem]:
        """Find similar semantic fact."""
        try:
            for item in self.semantic_memory:
                if (item.subject.lower() == subject.lower() and 
                    item.predicate.lower() == predicate.lower()):
                    return item
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar semantic fact: {e}")
            return None
    
    async def _find_similar_procedural_pattern(
        self,
        pattern_type: str,
        trigger_conditions: List[str]
    ) -> Optional[ProceduralMemoryItem]:
        """Find similar procedural pattern."""
        try:
            for item in self.procedural_memory:
                if item.pattern_type == pattern_type:
                    # Check trigger condition overlap
                    overlap = len(set(trigger_conditions) & set(item.trigger_conditions))
                    if overlap > len(trigger_conditions) * 0.6:  # 60% overlap threshold
                        return item
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar procedural pattern: {e}")
            return None
    
    async def _save_working_memory_item(self, item: WorkingMemoryItem) -> None:
        """Save working memory item to database."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO working_memory 
                (id, content, role, conversation_id, turn_number, timestamp, 
                 importance, access_count, last_accessed, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.content, item.role, item.conversation_id,
                item.turn_number, item.timestamp.isoformat(),
                item.importance, item.access_count, item.last_accessed.isoformat(),
                json.dumps(item.metadata), json.dumps(item.embedding)
            ))
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving working memory item: {e}")
    
    async def _save_episodic_memory_item(self, item: EpisodicMemoryItem) -> None:
        """Save episodic memory item to database."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO episodic_memory 
                (id, conversation_id, query, response, confidence, success, emotion,
                 decay_factor, timestamp, importance, access_count, last_accessed, 
                 metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.conversation_id, item.query, item.response,
                item.confidence, item.success, item.emotion, item.decay_factor,
                item.timestamp.isoformat(), item.importance, item.access_count,
                item.last_accessed.isoformat(), json.dumps(item.metadata),
                json.dumps(item.embedding)
            ))
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving episodic memory item: {e}")
    
    async def _save_semantic_memory_item(self, item: SemanticMemoryItem) -> None:
        """Save semantic memory item to database."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO semantic_memory 
                (id, fact_type, subject, predicate, object, confidence, source,
                 verification_status, timestamp, importance, access_count, 
                 last_accessed, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.fact_type, item.subject, item.predicate, item.object,
                item.confidence, item.source, item.verification_status,
                item.timestamp.isoformat(), item.importance, item.access_count,
                item.last_accessed.isoformat(), json.dumps(item.metadata),
                json.dumps(item.embedding)
            ))
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving semantic memory item: {e}")
    
    async def _save_procedural_memory_item(self, item: ProceduralMemoryItem) -> None:
        """Save procedural memory item to database."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO procedural_memory 
                (id, pattern_type, trigger_conditions, action_sequence, success_rate,
                 usage_count, context_constraints, timestamp, importance, 
                 access_count, last_accessed, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.pattern_type, json.dumps(item.trigger_conditions),
                json.dumps(item.action_sequence), item.success_rate, item.usage_count,
                json.dumps(item.context_constraints), item.timestamp.isoformat(),
                item.importance, item.access_count, item.last_accessed.isoformat(),
                json.dumps(item.metadata), json.dumps(item.embedding)
            ))
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving procedural memory item: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "semantic_memory_size": len(self.semantic_memory),
            "procedural_memory_size": len(self.procedural_memory),
            "total_access_count": self.access_count,
            "consolidation_count": self.consolidation_count,
            "last_consolidation": self.last_consolidation.isoformat(),
            "memory_capacities": {
                "working": self.working_memory_capacity,
                "episodic": self.episodic_memory_capacity,
                "semantic": self.semantic_memory_capacity,
                "procedural": self.procedural_memory_capacity
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        try:
            logger.info("Shutting down Integrated Memory System...")
            
            # Save all current memories
            await self._save_all_memories()
            
            # Close database connections
            if hasattr(self, 'sqlite_conn'):
                self.sqlite_conn.close()
            
            if hasattr(self, 'duck_conn'):
                self.duck_conn.close()
            
            logger.info("Integrated Memory System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during memory system shutdown: {e}")
            raise
    
    async def _save_all_memories(self) -> None:
        """Save all current memories to database."""
        try:
            # Save working memory
            for item in self.working_memory:
                await self._save_working_memory_item(item)
            
            # Save episodic memory
            for item in self.episodic_memory:
                await self._save_episodic_memory_item(item)
            
            # Save semantic memory
            for item in self.semantic_memory:
                await self._save_semantic_memory_item(item)
            
            # Save procedural memory
            for item in self.procedural_memory:
                await self._save_procedural_memory_item(item)
            
            logger.info("All memories saved to database")
            
        except Exception as e:
            logger.error(f"Error saving all memories: {e}")
            raise