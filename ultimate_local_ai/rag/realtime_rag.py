"""Real-Time RAG Pipeline with automated knowledge updates."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from ..utils.embeddings import EmbeddingManager
from ..utils.web_search import LocalWebSearch
from ..utils.knowledge_sources import KnowledgeSourceManager

logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    source: str
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""
    chunks: List[DocumentChunk]
    query: str
    total_results: int
    retrieval_time: float
    sources: List[str]
    is_current: bool = False
    needs_update: bool = False


class RealTimeRAG:
    """
    Real-Time Retrieval-Augmented Generation Pipeline.
    
    This system provides:
    - Semantic search through stored knowledge
    - Real-time web search when needed
    - Automated knowledge updates
    - Source prioritization and quality scoring
    - Continuous learning from user interactions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.get("vector_store_path", "./rag_store")
        )
        
        # Create collection with embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        
        try:
            self.collection = self.chroma_client.get_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function
            )
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(config)
        self.web_searcher = LocalWebSearch(config)
        self.knowledge_sources = KnowledgeSourceManager(config)
        
        # Background scheduler for automated updates
        self.scheduler = AsyncIOScheduler()
        
        # Settings
        self.max_chunk_size = config.get("max_chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.max_results = config.get("max_results", 10)
        self.freshness_threshold = config.get("freshness_threshold", 24)  # hours
        self.relevance_threshold = config.get("relevance_threshold", 0.7)
        
        # Statistics
        self.query_count = 0
        self.cache_hits = 0
        self.web_searches = 0
        self.total_documents = 0
        
        logger.info("Real-Time RAG Pipeline initialized")
    
    async def initialize(self) -> None:
        """Initialize the RAG pipeline."""
        try:
            logger.info("Initializing Real-Time RAG Pipeline...")
            
            # Initialize components
            await self.embedding_manager.initialize()
            await self.web_searcher.initialize()
            await self.knowledge_sources.initialize()
            
            # Setup automated knowledge updates
            await self._setup_automated_updates()
            
            # Start background scheduler
            self.scheduler.start()
            
            # Get initial stats
            try:
                self.total_documents = self.collection.count()
                logger.info(f"Loaded {self.total_documents} documents in knowledge base")
            except Exception as e:
                logger.warning(f"Could not get document count: {e}")
                self.total_documents = 0
            
            logger.info("Real-Time RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def retrieve_current_context(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_web_search: bool = True,
        force_refresh: bool = False
    ) -> RetrievalResult:
        """
        Retrieve the most current and relevant information for a query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            include_web_search: Whether to include web search if needed
            force_refresh: Force refresh of information
            
        Returns:
            RetrievalResult with relevant information
        """
        start_time = datetime.now()
        self.query_count += 1
        
        try:
            max_results = max_results or self.max_results
            
            # 1. Search stored knowledge
            stored_results = await self._search_stored_knowledge(query, max_results)
            
            # 2. Determine if we need current information
            needs_current_info = await self._needs_current_information(
                query, stored_results, force_refresh
            )
            
            # 3. Perform web search if needed
            if needs_current_info and include_web_search:
                logger.info(f"Performing web search for query: {query}")
                web_results = await self._perform_web_search(query, max_results // 2)
                
                # Index new information for future use
                await self._index_new_information(web_results)
                
                # Merge results
                combined_results = await self._merge_results(
                    stored_results, web_results, query
                )
                
                self.web_searches += 1
                
                return RetrievalResult(
                    chunks=combined_results,
                    query=query,
                    total_results=len(combined_results),
                    retrieval_time=(datetime.now() - start_time).total_seconds(),
                    sources=self._extract_sources(combined_results),
                    is_current=True,
                    needs_update=False
                )
            
            else:
                # Use stored results only
                self.cache_hits += 1
                
                return RetrievalResult(
                    chunks=stored_results,
                    query=query,
                    total_results=len(stored_results),
                    retrieval_time=(datetime.now() - start_time).total_seconds(),
                    sources=self._extract_sources(stored_results),
                    is_current=self._is_information_current(stored_results),
                    needs_update=False
                )
                
        except Exception as e:
            logger.error(f"Error in retrieve_current_context: {e}")
            
            # Return empty result on error
            return RetrievalResult(
                chunks=[],
                query=query,
                total_results=0,
                retrieval_time=(datetime.now() - start_time).total_seconds(),
                sources=[],
                is_current=False,
                needs_update=True
            )
    
    async def _search_stored_knowledge(
        self,
        query: str,
        max_results: int
    ) -> List[DocumentChunk]:
        """Search stored knowledge in ChromaDB."""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            
            if results and results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to relevance score (1 - distance for cosine similarity)
                    relevance_score = 1.0 - distance
                    
                    # Skip low-relevance results
                    if relevance_score < self.relevance_threshold:
                        continue
                    
                    chunk = DocumentChunk(
                        id=metadata.get("id", "unknown"),
                        content=doc,
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(
                            metadata.get("timestamp", datetime.now().isoformat())
                        ),
                        source=metadata.get("source", "unknown"),
                        relevance_score=relevance_score
                    )
                    
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching stored knowledge: {e}")
            return []
    
    async def _needs_current_information(
        self,
        query: str,
        stored_results: List[DocumentChunk],
        force_refresh: bool = False
    ) -> bool:
        """Determine if current information is needed."""
        if force_refresh:
            return True
        
        # Check for current-time keywords
        current_keywords = [
            "today", "latest", "recent", "current", "now", "2024", "2025",
            "news", "update", "what's new", "trending", "breaking"
        ]
        
        query_lower = query.lower()
        has_current_keywords = any(keyword in query_lower for keyword in current_keywords)
        
        # Check if stored results are fresh enough
        if stored_results:
            most_recent = max(stored_results, key=lambda x: x.timestamp)
            hours_old = (datetime.now() - most_recent.timestamp).total_seconds() / 3600
            
            if hours_old > self.freshness_threshold:
                return True
        
        # If no stored results and current keywords, definitely need web search
        if not stored_results and has_current_keywords:
            return True
        
        # Check if query is about recent events
        if has_current_keywords and len(stored_results) < 3:
            return True
        
        return False
    
    async def _perform_web_search(
        self,
        query: str,
        max_results: int
    ) -> List[DocumentChunk]:
        """Perform web search for current information."""
        try:
            # Use local web search
            search_results = await self.web_searcher.search(
                query=query,
                max_results=max_results,
                include_snippets=True
            )
            
            chunks = []
            
            for result in search_results:
                # Create document chunk from search result
                chunk = DocumentChunk(
                    id=f"web_{result.get('id', 'unknown')}",
                    content=result.get("content", result.get("snippet", "")),
                    metadata={
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source_type": "web_search",
                        "search_query": query,
                        "timestamp": datetime.now().isoformat()
                    },
                    timestamp=datetime.now(),
                    source=result.get("url", "web_search"),
                    relevance_score=result.get("score", 0.8)
                )
                
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    async def _index_new_information(
        self,
        chunks: List[DocumentChunk]
    ) -> None:
        """Index new information in ChromaDB for future use."""
        try:
            if not chunks:
                return
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.content)
                metadatas.append({
                    **chunk.metadata,
                    "indexed_at": datetime.now().isoformat(),
                    "relevance_score": chunk.relevance_score
                })
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            self.total_documents += len(chunks)
            logger.info(f"Indexed {len(chunks)} new documents")
            
        except Exception as e:
            logger.error(f"Error indexing new information: {e}")
    
    async def _merge_results(
        self,
        stored_results: List[DocumentChunk],
        web_results: List[DocumentChunk],
        query: str
    ) -> List[DocumentChunk]:
        """Merge and rank results from different sources."""
        try:
            all_results = stored_results + web_results
            
            # Remove duplicates based on content similarity
            unique_results = []
            seen_content = set()
            
            for result in all_results:
                content_hash = hash(result.content[:100])  # Use first 100 chars for deduplication
                
                if content_hash not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_hash)
            
            # Sort by relevance score and recency
            unique_results.sort(
                key=lambda x: (
                    x.relevance_score * 0.7 +  # Relevance weight
                    (1.0 if x.source.startswith("web_") else 0.5) * 0.3  # Recency weight
                ),
                reverse=True
            )
            
            return unique_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Error merging results: {e}")
            return stored_results + web_results
    
    def _extract_sources(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract unique sources from chunks."""
        sources = []
        
        for chunk in chunks:
            if chunk.source and chunk.source not in sources:
                sources.append(chunk.source)
        
        return sources
    
    def _is_information_current(self, chunks: List[DocumentChunk]) -> bool:
        """Check if information is current enough."""
        if not chunks:
            return False
        
        most_recent = max(chunks, key=lambda x: x.timestamp)
        hours_old = (datetime.now() - most_recent.timestamp).total_seconds() / 3600
        
        return hours_old <= self.freshness_threshold
    
    async def _setup_automated_updates(self) -> None:
        """Setup automated knowledge updates."""
        try:
            # News updates every hour
            self.scheduler.add_job(
                self._update_news_knowledge,
                "interval",
                hours=1,
                id="news_update",
                replace_existing=True
            )
            
            # Documentation updates daily
            self.scheduler.add_job(
                self._update_documentation_knowledge,
                "interval",
                days=1,
                id="docs_update",
                replace_existing=True
            )
            
            # Research papers updates every 6 hours
            self.scheduler.add_job(
                self._update_research_knowledge,
                "interval",
                hours=6,
                id="research_update",
                replace_existing=True
            )
            
            # Personal data sync every 5 minutes
            self.scheduler.add_job(
                self._sync_personal_data,
                "interval",
                minutes=5,
                id="personal_sync",
                replace_existing=True
            )
            
            # Cleanup old data weekly
            self.scheduler.add_job(
                self._cleanup_old_data,
                "interval",
                days=7,
                id="cleanup",
                replace_existing=True
            )
            
            logger.info("Automated knowledge updates scheduled")
            
        except Exception as e:
            logger.error(f"Error setting up automated updates: {e}")
    
    async def _update_news_knowledge(self) -> None:
        """Update knowledge base with latest news."""
        try:
            logger.info("Updating news knowledge...")
            
            # Get news from various sources
            news_updates = await self.knowledge_sources.get_news_updates()
            
            if news_updates:
                # Process and index news
                chunks = []
                for article in news_updates:
                    chunk = DocumentChunk(
                        id=f"news_{article['id']}",
                        content=article["content"],
                        metadata={
                            "title": article["title"],
                            "source": article["source"],
                            "url": article.get("url", ""),
                            "category": "news",
                            "timestamp": article["timestamp"]
                        },
                        timestamp=datetime.fromisoformat(article["timestamp"]),
                        source=article["source"],
                        relevance_score=0.8
                    )
                    chunks.append(chunk)
                
                await self._index_new_information(chunks)
                logger.info(f"Updated {len(chunks)} news articles")
            
        except Exception as e:
            logger.error(f"Error updating news knowledge: {e}")
    
    async def _update_documentation_knowledge(self) -> None:
        """Update knowledge base with latest documentation."""
        try:
            logger.info("Updating documentation knowledge...")
            
            # Get documentation updates
            doc_updates = await self.knowledge_sources.get_documentation_updates()
            
            if doc_updates:
                chunks = []
                for doc in doc_updates:
                    chunk = DocumentChunk(
                        id=f"doc_{doc['id']}",
                        content=doc["content"],
                        metadata={
                            "title": doc["title"],
                            "source": doc["source"],
                            "url": doc.get("url", ""),
                            "category": "documentation",
                            "timestamp": doc["timestamp"]
                        },
                        timestamp=datetime.fromisoformat(doc["timestamp"]),
                        source=doc["source"],
                        relevance_score=0.9
                    )
                    chunks.append(chunk)
                
                await self._index_new_information(chunks)
                logger.info(f"Updated {len(chunks)} documentation entries")
            
        except Exception as e:
            logger.error(f"Error updating documentation knowledge: {e}")
    
    async def _update_research_knowledge(self) -> None:
        """Update knowledge base with latest research."""
        try:
            logger.info("Updating research knowledge...")
            
            # Get research updates
            research_updates = await self.knowledge_sources.get_research_updates()
            
            if research_updates:
                chunks = []
                for paper in research_updates:
                    chunk = DocumentChunk(
                        id=f"research_{paper['id']}",
                        content=paper["content"],
                        metadata={
                            "title": paper["title"],
                            "authors": paper.get("authors", []),
                            "source": paper["source"],
                            "url": paper.get("url", ""),
                            "category": "research",
                            "timestamp": paper["timestamp"]
                        },
                        timestamp=datetime.fromisoformat(paper["timestamp"]),
                        source=paper["source"],
                        relevance_score=0.85
                    )
                    chunks.append(chunk)
                
                await self._index_new_information(chunks)
                logger.info(f"Updated {len(chunks)} research papers")
            
        except Exception as e:
            logger.error(f"Error updating research knowledge: {e}")
    
    async def _sync_personal_data(self) -> None:
        """Sync personal data and recent interactions."""
        try:
            # Get personal data updates
            personal_updates = await self.knowledge_sources.get_personal_updates()
            
            if personal_updates:
                chunks = []
                for item in personal_updates:
                    chunk = DocumentChunk(
                        id=f"personal_{item['id']}",
                        content=item["content"],
                        metadata={
                            "type": item["type"],
                            "source": "personal",
                            "category": "personal",
                            "timestamp": item["timestamp"]
                        },
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        source="personal",
                        relevance_score=1.0
                    )
                    chunks.append(chunk)
                
                await self._index_new_information(chunks)
            
        except Exception as e:
            logger.error(f"Error syncing personal data: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old and irrelevant data."""
        try:
            logger.info("Cleaning up old data...")
            
            # This would implement cleanup logic
            # For now, just log the action
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        source: str = "manual"
    ) -> str:
        """Add a document to the knowledge base."""
        try:
            # Generate unique ID
            doc_id = f"manual_{datetime.now().timestamp()}"
            
            # Create document chunk
            chunk = DocumentChunk(
                id=doc_id,
                content=content,
                metadata={
                    **metadata,
                    "source": source,
                    "category": "manual",
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source=source,
                relevance_score=0.9
            )
            
            # Index the document
            await self._index_new_information([chunk])
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Update an existing document."""
        try:
            # Delete old version
            self.collection.delete(ids=[doc_id])
            
            # Add updated version
            await self.add_document(content, metadata, metadata.get("source", "manual"))
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from the knowledge base."""
        try:
            self.collection.delete(ids=[doc_id])
            self.total_documents -= 1
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics."""
        return {
            "total_documents": self.total_documents,
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits,
            "web_searches": self.web_searches,
            "cache_hit_rate": self.cache_hits / self.query_count if self.query_count > 0 else 0,
            "freshness_threshold_hours": self.freshness_threshold,
            "relevance_threshold": self.relevance_threshold
        }
    
    async def shutdown(self) -> None:
        """Shutdown the RAG pipeline."""
        try:
            logger.info("Shutting down Real-Time RAG Pipeline...")
            
            # Stop scheduler
            if self.scheduler.running:
                self.scheduler.shutdown()
            
            # Shutdown components
            await self.web_searcher.shutdown()
            await self.knowledge_sources.shutdown()
            
            logger.info("Real-Time RAG Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during RAG shutdown: {e}")
            raise