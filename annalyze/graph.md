⏺ Perfect! Bây giờ tôi có thể phân tích chi tiết cách RAGFlow thực hiện Graph RAG search:

  RAGFlow Graph RAG Search - Chi Tiết Implementation

  1. Knowledge Graph Storage Structure trong Elasticsearch

  RAGFlow lưu trữ graph data trong Elasticsearch với các loại documents khác nhau:

  # Phân loại documents trong ES:
  "knowledge_graph_kwd": "entity"      # Entity documents
  "knowledge_graph_kwd": "relation"    # Relationship documents  
  "knowledge_graph_kwd": "subgraph"    # Community/subgraph documents
  "knowledge_graph_kwd": "graph"       # Full graph metadata

  Entity Documents:

  {
    "_source": {
      "knowledge_graph_kwd": "entity",
      "entity_kwd": "Neural Networks",
      "entity_type_kwd": "technology",
      "content_with_weight": "Neural networks are computational models...",
      "rank_flt": 0.85,                 # PageRank score
      "n_hop_with_weight": [            # N-hop neighbors với weights
        {
          "path": ["Neural Networks", "Machine Learning", "AI"],
          "weights": [0.8, 0.7, 0.6]
        }
      ]
    }
  }

  Relationship Documents:

  {
    "_source": {
      "knowledge_graph_kwd": "relation",
      "from_entity_kwd": "Neural Networks",
      "to_entity_kwd": "Deep Learning",
      "content_with_weight": "Neural networks are the foundation of deep learning...",
      "weight_int": 8                   # Relationship strength
    }
  }

  2. Graph RAG Search Flow

  Step 1: Query Analysis & Entity Extraction

  # graphrag/search.py:160 - query_rewrite()
  def query_rewrite(self, llm, question, idxnms, kb_ids):
      # Get entity type samples từ existing graph
      ty2ents = get_entity_type2samples(idxnms, kb_ids)

      # LLM prompt để extract entities và types từ query
      hint_prompt = PROMPTS["minirag_query2kwd"].format(
          query=question,
          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2)
      )

      result = self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {})

      # Parse JSON response
      keywords_data = json_repair.loads(result)
      type_keywords = keywords_data.get("answer_type_keywords", [])    # Entity types
      entities_from_query = keywords_data.get("entities_from_query", [])[:5]  # Specific entities

      return type_keywords, entities_from_query

  Step 2: Multi-dimensional Entity Retrieval

  # graphrag/search.py:167-168
  # A. Entity retrieval by keywords
  ents_from_query = self.get_relevant_ents_by_keywords(
      ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold
  )

  # B. Entity retrieval by types  
  ents_from_types = self.get_relevant_ents_by_types(
      ty_kwds, filters, idxnms, kb_ids, 10000
  )

  A. Keyword-based Entity Search:

  # graphrag/search.py:106-115
  def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, 
  N=56):
      filters = deepcopy(filters)
      filters["knowledge_graph_kwd"] = "entity"  # Only search entity documents

      # Vector similarity search trên entity embeddings
      matchDense = self.get_vector(", ".join(keywords), emb_mdl, 1024, sim_thr)

      es_res = self.dataStore.search(
          ["content_with_weight", "entity_kwd", "rank_flt"],
          [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids
      )

      return self._ent_info_from_(es_res, sim_thr)  # Parse entity results

  B. Type-based Entity Search:

  # graphrag/search.py:128-138
  def get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
      filters = deepcopy(filters)
      filters["knowledge_graph_kwd"] = "entity"
      filters["entity_type_kwd"] = types        # Filter by entity types

      ordr = OrderByExpr()
      ordr.desc("rank_flt")                     # Order by PageRank desc

      es_res = self.dataStore.search(
          ["entity_kwd", "rank_flt"], [], filters, [], ordr, 0, N, idxnms, kb_ids
      )

      return self._ent_info_from_(es_res, 0)

  Step 3: Relationship Retrieval

  # graphrag/search.py:169
  rels_from_txt = self.get_relevant_relations_by_txt(
      qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold
  )

  def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
      filters = deepcopy(filters)
      filters["knowledge_graph_kwd"] = "relation"  # Only search relation documents

      # Vector similarity search trên relationship descriptions
      matchDense = self.get_vector(txt, emb_mdl, 1024, sim_thr)

      es_res = self.dataStore.search(
          ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"],
          [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids
      )

      return self._relation_info_from_(es_res, sim_thr)

  Step 4: N-hop Path Analysis

  # graphrag/search.py:170-186
  nhop_pathes = defaultdict(dict)
  for _, ent in ents_from_query.items():
      nhops = ent.get("n_hop_ents", [])        # Pre-computed n-hop paths

      for nbr in nhops:
          path = nbr["path"]                    # ["A", "B", "C"] - path sequence
          wts = nbr["weights"]                  # [0.8, 0.7, 0.6] - edge weights

          # Extract pairwise relationships từ path
          for i in range(len(path) - 1):
              f, t = path[i], path[i + 1]
              if (f, t) in nhop_pathes:
                  # Accumulate similarity với distance decay
                  nhop_pathes[(f, t)]["sim"] += ent["sim"] / (2 + i)
              else:
                  nhop_pathes[(f, t)]["sim"] = ent["sim"] / (2 + i)
              nhop_pathes[(f, t)]["pagerank"] = wts[i]

  Step 5: Graph-based Scoring & Ranking

  # graphrag/search.py:192-196
  # Boost entities that match both keywords và types
  for ent in ents_from_types.keys():
      if ent not in ents_from_query:
          continue
      ents_from_query[ent]["sim"] *= 2        # Double the similarity score

  3. Graph Search Integration với Regular Search

  Filter Integration:

  # rag/nlp/search.py:64-66
  # RAGFlow search supports graph filters:
  for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd",
  "to_entity_kwd", "removed_kwd"]:
      if key in req and req[key] is not None:
          condition[key] = req[key]

  Search Request với Graph Filters:

  # Example search request:
  req = {
      "question": "How do neural networks work?",
      "kb_ids": ["kb_123"],
      "entity_kwd": ["Neural Networks", "Deep Learning"],     # Specific entities
      "from_entity_kwd": ["Neural Networks"],                 # Source entities  
      "to_entity_kwd": ["Machine Learning"],                  # Target entities
      "knowledge_graph_kwd": ["entity", "relation"]           # Graph data types
  }

  4. Complete Graph RAG Query Example

  # User query: "How do neural networks enable deep learning?"

  # Step 1: Query analysis
  ty_kwds, ents = query_rewrite(llm, query, index_names, kb_ids)
  # → ty_kwds: ["technology", "algorithm"]  
  # → ents: ["neural networks", "deep learning"]

  # Step 2: Multi-dimensional entity retrieval
  # A. Keyword-based: Search entities similar to "neural networks, deep learning"
  ents_from_query = {
      "Neural Networks": {"sim": 0.95, "pagerank": 0.85, "description": "..."},
      "Deep Learning": {"sim": 0.88, "pagerank": 0.78, "description": "..."},
      "Backpropagation": {"sim": 0.72, "pagerank": 0.65, "description": "..."}
  }

  # B. Type-based: Get top entities của types ["technology", "algorithm"] 
  ents_from_types = {
      "Convolutional Neural Networks": {"sim": 0.82, "pagerank": 0.75},
      "Gradient Descent": {"sim": 0.78, "pagerank": 0.70}
  }

  # Step 3: Relationship retrieval
  rels_from_txt = {
      ("Neural Networks", "Deep Learning"): {"sim": 0.90, "pagerank": 8, "description": "..."},
      ("Deep Learning", "Machine Learning"): {"sim": 0.85, "pagerank": 7, "description": "..."}
  }

  # Step 4: N-hop analysis từ pre-computed paths
  nhop_pathes = {
      ("Neural Networks", "Backpropagation"): {"sim": 0.75, "pagerank": 0.68},
      ("Backpropagation", "Gradient Descent"): {"sim": 0.65, "pagerank": 0.60}
  }

  # Step 5: Combined ranking 
  # Entities that match both keywords và types get 2x boost
  # Final context construction từ top-ranked entities và relationships

  5. Key Advantages của RAGFlow Graph Search

  1. Elasticsearch Integration: Leverages ES's powerful indexing và search capabilities
  2. Multi-dimensional Retrieval: Keywords, types, relationships, n-hop paths
  3. Pre-computed N-hops: Avoids expensive graph traversal at query time
  4. Hybrid Scoring: Combines similarity, PageRank, và graph structure
  5. Scalable: Works với large knowledge graphs through ES sharding
  6. Flexible Filtering: Can filter by specific entities, types, relationships

  6. Performance Characteristics

  - Pre-indexing: All graph data indexed trong ES upfront
  - Query Time: Fast ES queries thay vì graph traversal
  - Memory Efficient: Graph structure stored trong ES, not in memory
  - Scalable: Horizontal scaling through ES clusters
  - Cached: LLM query analysis results được cached để improve performance

  RAGFlow's Graph RAG search is essentially "Graph-aware Elasticsearch Search" - combining
  traditional search efficiency với graph reasoning capabilities!