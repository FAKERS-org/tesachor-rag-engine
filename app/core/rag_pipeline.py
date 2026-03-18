"""
RAG Pipeline Architecture:
Query → Preprocessing → Retrieval (Vector+Keyword) → Reranking → 
Context Building → Prompt Engineering → Generation → Post-processing
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import json
from datetime import datetime
from app.config import settings

@dataclass
class RetrievedChunk:
    """Structured retrieval result"""
    content: str
    metadata: Dict
    score: float
    chunk_id: str
    source_type: str  # "attraction", "restaurant", "event"

@dataclass
class RAGResponse:
    """Final structured output"""
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    generation_time_ms: float
    retrieval_time_ms: float
    total_tokens: int
    confidence_score: float

class RAGPipeline:
    """
    Production RAG Pipeline for Phnom Penh Tourism
    Implements: Hybrid Retrieval + Reranking + Structured Generation
    """
    
    def __init__(
        self,
        vector_store,
        llm_client,
        embedding_client,
        cache_client=None,
        reranker=None
    ):
        self.vector_store = vector_store      # Chroma/Weaviate
        self.llm = llm_client                 # GPT-4 / Claude
        self.embeddings = embedding_client    # text-embedding-3
        self.cache = cache_client             # Redis (optional)
        self.reranker = reranker              # Cross-encoder (optional)
        
        # Metrics tracking
        self.query_count = 0
    
    async def query(
        self,
        question: str,
        filters: Optional[Dict] = None,
        conversation_context: Optional[List[Dict]] = None,
        top_k: int = None
    ) -> RAGResponse:
        """
        Main entry point: End-to-end RAG processing
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        # Step 1: Query Understanding (Preprocessing)
        enhanced_query = self._enhance_query(question, conversation_context)
        
        # Step 2: Check Cache
        cache_key = self._generate_cache_key(enhanced_query, filters)
        if settings.ENABLE_CACHE and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return RAGResponse(**cached)
        
        # Step 3: RETRIEVAL (The "R" in RAG)
        retrieval_start = time.perf_counter()
        candidates = await self._retrieve(enhanced_query, filters, top_k * 2)  # Over-fetch for reranking
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        # Step 4: RERANKING (Boost relevance precision)
        if getattr(settings, "RERANK_ENABLED", False) and self.reranker:
            candidates = await self._rerank(enhanced_query, candidates, top_k)
        else:
            candidates = candidates[:top_k]
        
        # Step 5: Confidence Scoring
        confidence = self._calculate_confidence(candidates)
        
        # Step 6: GENERATION (The "G" in RAG)
        generation_start = time.perf_counter()
        answer, tokens = await self._generate(
            question=question,
            retrieved_chunks=candidates,
            conversation_context=conversation_context
        )
        generation_time = (time.perf_counter() - generation_start) * 1000
        
        # Step 7: Build Response
        response = RAGResponse(
            answer=answer,
            retrieved_chunks=candidates,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_tokens=tokens,
            confidence_score=confidence
        )
        
        # Step 8: Cache Result
        if settings.ENABLE_CACHE and self.cache and confidence > 0.6:
            await self.cache.set(cache_key, response.__dict__, ttl=settings.CACHE_TTL)
        
        self.query_count += 1
        return response
    
    def _enhance_query(
        self,
        query: str,
        conversation_context: Optional[List[Dict]]
    ) -> str:
        """
        Query preprocessing: Expand tourism-specific terms
        Example: "palace" → "Royal Palace Cambodia Phnom Penh"
        """
        # Add temporal context (your system date: 2026-03-17)
        enhanced = query
        
        # Conversation context injection (for follow-up questions)
        if conversation_context and len(conversation_context) > 0:
            last_topic = conversation_context[-1].get("topic", "")
            if last_topic and self._is_follow_up(query):
                enhanced = f"{last_topic} {query}"
        
        # Tourism domain expansion
        expansions = {
            "palace": "Royal Palace Cambodia Phnom Penh",
            "killing fields": "Choeung Ek Genocidal Center Khmer Rouge",
            "market": "Russian Market Central Market Phsar Thmey",
            "temple": "Wat Phnom Wat Ounalom pagoda",
            "river": "Tonle Sap Mekong Sisowath Quay Riverside",
        }
        
        for term, expansion in expansions.items():
            if term.lower() in query.lower():
                enhanced = f"{enhanced} {expansion}"
        
        return enhanced
    
    async def _retrieve(
        self,
        query: str,
        filters: Optional[Dict],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        HYBRID RETRIEVAL: Vector Similarity + Keyword BM25
        This is crucial for production RAG - combines semantic + lexical search
        """
        chunks = []
        
        # A) Vector Search (Semantic meaning)
        vector_results = await self.vector_store.asimilarity_search_with_score(
            query=query,
            k=top_k,
            filter=filters
        )
        
        for doc, score in vector_results:
            chunks.append(RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                source_type=doc.metadata.get("type", "attraction")
            ))
        
        # B) Keyword Search (if hybrid enabled)
        if getattr(settings, "HYBRID_SEARCH", False):
            keyword_results = await self._keyword_search(query, filters, top_k)
            
            # Merge and deduplicate (Reciprocal Rank Fusion)
            chunks = self._fuse_results(chunks, keyword_results)
        
        return chunks
    
    async def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        BM25/TF-IDF keyword search for exact term matching
        Useful for: names, specific locations, "free entry"
        """
        # Implementation depends on vector store (Chroma supports BM25 via FullTextSearch)
        # This is a simplified version
        return []  # Placeholder - implement based on your vector DB
    
    def _fuse_results(
        self,
        vector_chunks: List[RetrievedChunk],
        keyword_chunks: List[RetrievedChunk],
        k: int = 60  # RRF constant
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion: Combine vector and keyword results
        Formula: score = Σ 1/(k + rank) for each list containing the item
        """
        scores = {}
        chunk_map = {}
        
        # Score vector results
        for rank, chunk in enumerate(vector_chunks):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_map[key] = chunk
        
        # Score keyword results
        for rank, chunk in enumerate(keyword_chunks):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk
        
        # Sort by fused score
        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda x: scores.get(x.chunk_id, 0),
            reverse=True
        )
        
        return sorted_chunks
    
    async def _rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int
    ) -> List[RetrievedChunk]:
        """
        Cross-encoder reranking: More precise relevance scoring
        Uses a small transformer (e.g., BAAI/bge-reranker-base) to score query-doc pairs
        """
        if not self.reranker or len(candidates) <= top_n:
            return candidates
        
        pairs = [(query, c.content) for c in candidates]
        scores = await self.reranker.apredict(pairs)
        
        # Sort by reranker score
        scored_candidates = [
            (chunk, score) for chunk, score in zip(candidates, scores)
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored_candidates[:top_n]]
    
    async def _generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_context: Optional[List[Dict]]
    ) -> Tuple[str, int]:
        """
        Structured Generation with Tourism-Specific Prompting
        """
        # Build context string with citations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.metadata
            context_parts.append(f"""
[Source {i}] {meta.get('name', 'Unknown')}
Type: {chunk.source_type}
Location: {meta.get('location', 'N/A')}
Entry: {meta.get('entry_fee', 'Free')}
Hours: {meta.get('opening_hours', 'Unknown')}
Content: {chunk.content}
""")
        
        context_str = "\n---\n".join(context_parts)
        
        # Conversation history (last 3 exchanges)
        history_str = ""
        if conversation_context:
            history_str = "Previous conversation:\n"
            for ex in conversation_context[-3:]:
                history_str += f"User: {ex['user']}\nAssistant: {ex['bot']}\n"
        
        # Tourism-optimized prompt
        today = datetime.now().strftime("%B %d, %Y")
        system_prompt = f"""You are an expert Phnom Penh tourism guide with deep knowledge of:
- Cambodian history (Khmer Empire, French colonial, Khmer Rouge era)
- Local culture and etiquette (temple dress codes, bargaining, tuk-tuk negotiations)
- Practical logistics (opening hours, entry fees, location proximity)
- Food safety and dietary considerations

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Cite sources using [Source 1], [Source 2] format
3. Include practical details: prices, hours, dress codes, best times to visit
4. For historical sites (Killing Fields, Tuol Sleng), be respectful and factual
5. Suggest nearby attractions if relevant
6. If information is missing, say so clearly

Current date: {today} (consider seasonal factors)"""

        user_prompt = f"""{history_str}

Context from Phnom Penh tourism database:
{context_str}

User question: {question}

Provide a helpful, structured answer:"""
        
        # Call LLM
        response = await self.llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        total_tokens = getattr(
            response,
            "tokens",
            len(system_prompt.split()) + len(user_prompt.split()) + len(response.content.split())
        )
        
        return response.content, total_tokens
    
    def _calculate_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """
        Confidence score based on retrieval quality
        """
        if not chunks:
            return 0.0
        
        # Chroma returns distance scores (lower is better): relevance = 1 / (1 + distance)
        avg_distance = sum(c.score for c in chunks) / len(chunks)
        normalized = 1 / (1 + max(avg_distance, 0))
        return round(normalized, 2)
    
    def _is_follow_up(self, query: str) -> bool:
        """Detect if query is a follow-up question"""
        follow_up_indicators = ["it", "there", "that place", "the museum", "how much", "what about"]
        return any(ind in query.lower() for ind in follow_up_indicators)
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict]) -> str:
        """Generate deterministic cache key"""
        key_data = f"{query}:{json.dumps(filters, sort_keys=True)}"
        import hashlib
        return f"rag:{hashlib.md5(key_data.encode()).hexdigest()}"

