
  Dựa trên phân tích code, đây là những khác biệt chính trong cách approach search:

  1. Query Processing Pipeline

  RAGFlow:

  - Multi-stage processing: Query normalization → Term weighting → Synonym expansion → Field boosting
  - Language-aware: Different tokenization cho Chinese/English
  - Smart preprocessing: Remove filler words, handle mixed languages
  # rag/nlp/query.py:85
  def question(self, txt, tbl="qa", min_match: float = 0.6):
      txt = FulltextQueryer.add_space_between_eng_zh(txt)
      txt = re.sub(r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+", " ",
                   rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())))
      # Term weighting + synonym expansion + field boosting

  LightRAG:

  - Simpler approach: Direct embedding + basic text processing
  - Entity-focused: Extract entities from query for graph traversal
  - Graph-first: Query understanding through knowledge graph

  2. Retrieval Strategy

  RAGFlow - Hybrid Search với Intelligent Fusion:

  # rag/nlp/search.py:114
  fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
  matchExprs = [matchText, matchDense, fusionExpr]
  - Dense retrieval: Vector similarity (95% weight)
  - Sparse retrieval: Full-text search (5% weight)
  - Field-specific boosting: title^10, important^30, content^2
  - Fallback mechanism: Lower thresholds when no results

  LightRAG - Graph-enhanced Retrieval:

  - Entity-driven: Start from query entities
  - Graph traversal: Follow relationships for context expansion
  - Community-based: Retrieve from entity communities
  - Simpler vector: Basic embedding similarity

  3. Index Architecture

  RAGFlow:

  # Multiple specialized indexes
  query_fields = [
      "title_tks^10",           # Title tokens với weight 10x
      "title_sm_tks^5",         # Title small tokens 5x
      "important_kwd^30",       # Important keywords 30x
      "important_tks^20",       # Important tokens 20x  
      "question_tks^20",        # Question tokens 20x
      "content_ltks^2",         # Content large tokens 2x
      "content_sm_ltks",        # Content small tokens 1x
  ]
  - Multi-field indexing: 7+ specialized fields với different weights
  - Hierarchical tokenization: Large + small tokens for different granularity
  - Rich metadata: Position, page numbers, importance scores

  LightRAG:

  - Simpler structure: Text chunks + entity graph
  - Graph-centric: Entities và relationships là first-class citizens
  - Unified approach: Same retrieval mechanism cho text và graph

  4. Re-ranking & Post-processing

  RAGFlow:

  # Complex re-ranking với multiple signals
  highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")
  aggs = self.dataStore.getAggregation(res, "docnm_kwd")
  # PageRank, relevance scores, position weights
  - Multi-signal ranking: PageRank + relevance + position + recency
  - Highlight generation: Intelligent snippet extraction
  - Aggregation: Group results by documents
  - Rich result format: Scores, positions, metadata

  LightRAG:

  - Simpler ranking: Mainly similarity-based
  - Graph influence: Entity importance affects ranking
  - Direct results: Less post-processing

  5. Performance & Scalability

  RAGFlow:

  # task_executor.py - Enterprise-grade processing
  async def handle_task():
      # Distributed task processing
      # Redis queuing
      # Worker pools
      # Health monitoring
  - Distributed architecture: Multiple workers, Redis coordination
  - Batch processing: Configurable batch sizes
  - Caching layers: Redis cho LLM responses, embeddings
  - Resource management: Connection pools, memory monitoring

  LightRAG:

  - Single-process: Simpler deployment model
  - In-memory: Local processing, less complex infrastructure
  - Research-focused: Optimization for experimentation

  6. Câu trả lời cụ thể: "Khi search, RAGFlow làm gì khác LightRAG?"

  RAGFlow's Search Flow:

  1. Query Analysis: Tokenize → Remove stop words → Extract terms → Weight by importance
  2. Multi-field Search: Search across 7 specialized fields với different boosts
  3. Hybrid Retrieval: Combine text search (5%) + vector search (95%)
  4. Intelligent Fallback: Lower thresholds if no results
  5. Rich Re-ranking: PageRank + position + relevance + recency
  6. Post-processing: Highlight generation + aggregation + metadata enrichment

  LightRAG's Search Flow:

  1. Entity Extraction: Identify entities in query
  2. Graph Lookup: Find entities in knowledge graph
  3. Community Retrieval: Get documents from entity communities
  4. Vector Search: Supplement với embedding similarity
  5. Simple Ranking: Mainly similarity-based

  Kết luận:

  RAGFlow approach search như một enterprise search engine với:
  - Sophisticated query understanding
  - Multi-signal retrieval
  - Complex re-ranking
  - Production-ready infrastructure

  LightRAG approach search như một research tool với:
  - Graph-first thinking
  - Entity-driven retrieval
  - Simpler but effective mechanisms
  - Focus on knowledge connections

  RAGFlow phù hợp cho production systems cần high precision và rich features, trong khi LightRAG phù
  hợp cho research và rapid prototyping với focus on knowledge graph reasoning.