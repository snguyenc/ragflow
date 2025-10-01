Graph RAG Integration ✅

  RAGFlow tích hợp Graph RAG rất toàn diện:

  Knowledge Graph Construction:

  - Entity Extraction: Sử dụng LLM với predefined entity types (person, organization, geo, event,
  category)
  - Relationship Extraction: NetworkX-based graph với entity relationships
  - Multi-gleaning: Up to 2 iterations để improve extraction quality
  - Prompt-based: Structured prompts cho consistent extraction

  Graph Processing:

  - NetworkX Integration: Full graph analysis capabilities
  - Entity Resolution: Deduplication và merging similar entities
  - Community Detection: Hierarchical clustering cho document organization
  - Graph Embedding: Vector representations cho graph nodes

  Query Enhancement:

  - Graph Traversal: Follow entity relationships cho context expansion
  - Community-based Retrieval: Retrieve from related document clusters
  - Entity Filtering: Search by specific entities (entity_kwd, from_entity_kwd, to_entity_kwd)

  5. Advanced Features 🔄

  RAPTOR (Recursive Abstractive Processing):

  - Hierarchical Clustering: UMAP + Gaussian Mixture Models
  - Tree Organization: Multi-level document summarization
  - Recursive Processing: Build abstract summaries at different levels
  - Optimal Clustering: Adaptive số lượng clusters

  Multimodal Capabilities:

  - Vision Integration: OCR, layout analysis, table detection
  - Image Processing: PIL-based image concatenation và cropping
  - Mixed Content: Text + image trong cùng chunks
  - Visual Question Answering: Support cho image-based queries

  Cross-language Support:

  - Tokenization: Different strategies cho Chinese/English
  - Traditional/Simplified Chinese: Auto conversion
  - Mixed Language: Chinese-English query processing
  - Cultural Context: Support cho cultural-specific patt

⏺ Tuyệt vời! Bây giờ tôi hiểu rõ hệ thống metadata phức tạp của RAGFlow. Đây là phân tích chi tiết:

  RAGFlow Metadata System: Tags & Auto Question Generation

  1. TAG System Architecture

  TAG_FLD (tag_feas) - Auto Tagging:

  # rag/settings.py:64
  TAG_FLD = "tag_feas"  # Stored as JSON: {"tag1": relevance_score, "tag2": score}

  Quá trình Auto Tagging:

  1. Collect All Tags: Aggregate từ toàn bộ knowledge base
  # search.py:470-473
  res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids,
  ["tag_kwd"])
  res = self.dataStore.getAggregation(res, "tag_kwd")
  total = np.sum([c for _, c in res])
  return {t: (c + 1) / (total + S) for t, c in res}  # Bayesian smoothing

  2. Content-based Tag Prediction:
  # search.py:475-486
  def tag_content(self, tenant_id, kb_ids, doc, all_tags, topn_tags=3):
      # Search similar content
      match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"])
      res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids,
  ["tag_kwd"])

      # Calculate TF-IDF like scoring với Bayesian smoothing
      tag_fea = sorted([(a, 0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001))))
                       for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]

      doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}

  3. LLM-based Auto Tagging:
  # prompts.py:254-293
  def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
      # Sử dụng few-shot learning với examples
      template = PROMPT_JINJA_ENV.from_string(CONTENT_TAGGING_PROMPT_TEMPLATE)
      # LLM generates JSON: {"tag1": relevance_1_to_10, "tag2": relevance}

  2. QUESTION System (question_kwd, question_tks)

  Auto Question Generation:

  # prompts.py:185-197
  def question_proposal(chat_mdl, content, topn=3):
      template = PROMPT_JINJA_ENV.from_string(QUESTION_PROMPT_TEMPLATE)
      # LLM generates potential questions người dùng có thể hỏi về content này

  Question Indexing:

  # Stored in multiple fields:
  d["question_kwd"] = [list of questions]  # Raw questions
  d["question_tks"] = tokenized_questions  # Tokenized for search

  3. PAGERANK System

  # rag/settings.py:63
  PAGERANK_FLD = "pagerank_fea"  # Document importance score

  - Graph-based scoring: Dựa trên entity relationships trong Graph RAG
  - Document authority: Similar to web PageRank
  - User-defined: Có thể manually set pagerank cho documents

  4. Ý nghĩa trong Search Process

  Multi-dimensional Ranking:

  # search.py:257-270
  pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))

  # Tag-based query expansion
  q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
  for i in search_res.ids:
      if not search_res.field[i].get(TAG_FLD):
          continue
      # Calculate tag similarity với query
      for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
          if t in query_rfea:
              scores[i] += sc * query_rfea[t] / q_denor

  Query Processing Enhancement:

  1. Tag Query Expansion:
  # search.py:488-501
  def tag_query(self, question, tenant_ids, kb_ids, all_tags, topn_tags=3):
      # Extract tags từ user query
      # Expand query với related tags từ knowledge base
      return {tag: relevance_score}  # Used for ranking boost

  2. Question Matching:
  # query.py:30-38 - Field boosting
  query_fields = [
      "title_tks^10",
      "question_tks^20",    # Questions có weight cao (20x)
      "important_kwd^30",
      "content_ltks^2",
  ]

  5. So sánh với LightRAG

  | Aspect            | RAGFlow Metadata                                            | LightRAG
           |
  |-------------------|-------------------------------------------------------------|----------------
  ---------|
  | Tags              | Auto LLM tagging + TF-IDF scoring + few-shot learning       | Basic entity
  extraction |
  | Questions         | Auto generation + search optimization                       | Không có
           |
  | Ranking           | Multi-signal: PageRank + tag relevance + position + recency | Simple
  similarity       |
  | Query Enhancement | Tag expansion + question matching + field boosting          | Entity-based
  expansion  |
  | Learning          | Few-shot examples + Bayesian smoothing                      | Static approach
           |

  6. Practical Impact

  Search Quality Enhancement:

  1. Semantic Tags: Auto-generated tags capture document themes
  2. Question Anticipation: Pre-generated questions match user intent
  3. Authority Scoring: PageRank identifies important documents
  4. Multi-field Optimization: Different weights for different content types

  Example Search Scenario:

  User Query: "machine learning algorithms"

  RAGFlow Process:
  1. Extract tags from query: ["machine_learning", "algorithms", "AI"]  
  2. Boost documents với matching tags (weighted by relevance scores)
  3. Match pre-generated questions: "What are ML algorithms?", "How do ML models work?"
  4. Apply PageRank cho authoritative documents
  5. Multi-field search với question_tks^20 boost

  Result: Highly relevant, authority documents về ML algorithms

  Kết luận:

  RAGFlow's metadata system tạo ra một "semantic index" rất phong phú:
  - Tags: Capture document semantics và themes
  - Questions: Bridge user intent với document content
  - PageRank: Authority và importance scoring
  - Multi-field: Optimize cho different query types

  Điều này cho phép RAGFlow achieve much higher search precision so với simple vector similarity, đặc
   biệt trong enterprise environments với large, diverse document collections.


   Tuyệt! Bây giờ tôi thấy rõ cách RAGFlow sử dụng metadata trong cả search process và reranking. Đây
  là phân tích chi tiết:

  RAGFlow Metadata Usage: Search Process + Reranking

  1. TRONG SEARCH PROCESS (Before Retrieval)

  Query-time Tag Analysis:

  # rag/app/tag.py:124-147
  def label_question(question, kbs):
      # Extract tags từ user question
      tags = settings.retrievaler.tag_query(question, tenant_ids, tag_kb_ids, all_tags, topn_tags=3)
      return tags  # {"tag1": relevance_score, "tag2": score}

  Multi-field Search với Metadata Fields:

  # query.py:30-38
  self.query_fields = [
      "title_tks^10",
      "title_sm_tks^5",
      "important_kwd^30",        # Metadata field với 30x boost
      "important_tks^20",        # Metadata field với 20x boost
      "question_tks^20",         # Auto-generated questions với 20x boost
      "content_ltks^2",
      "content_sm_ltks",
  ]

  Rank Features trong Search Query:

  # es_conn.py:215-219 - NGAY TRONG ELASTICSEARCH QUERY
  if bqry and rank_feature:
      for fld, sc in rank_feature.items():
          if fld != PAGERANK_FLD:
              fld = f"{TAG_FLD}.{fld}"  # tag_feas.machine_learning
          bqry.should.append(Q("rank_feature", field=fld, linear={}, boost=sc))

  Ý nghĩa: Elasticsearch sẽ boost documents có matching tags ngay trong search process, không phải
  sau khi search xong!

  2. TRONG RETRIEVAL PROCESS (During Search)

  Hybrid Search với Metadata Boosting:

  # search.py:114-118
  fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
  matchExprs = [matchText, matchDense, fusionExpr]

  res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy,
                             offset, limit, idx_names, kb_ids,
                             rank_feature=rank_feature)  # ← Tags được pass vào search

  Real Search Call với Tag Boosting:

  # Khi user search "machine learning algorithms":

  # 1. Extract tags từ query
  labels = label_question("machine learning algorithms", kbs)
  # → {"machine_learning": 8, "algorithms": 7, "AI": 5}

  # 2. Pass vào search engine
  res = dealer.search(req, idx_names, kb_ids, emb_mdl, highlight=False,
                      rank_feature=labels)  # ← Boost documents có matching tags

  3. POST-SEARCH RERANKING (After Initial Results)

  Tag-based Score Adjustment:

  # search.py:257-270
  pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))

  # Calculate tag similarity với query tags
  q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
  for i in search_res.ids:
      if not search_res.field[i].get(TAG_FLD):
          continue
      # Add tag relevance score
      for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
          if t in query_rfea:
              scores[i] += sc * query_rfea[t] / q_denor

  Multi-signal Final Ranking:

  # Combined ranking factors:
  final_score = (
      base_search_score +           # Text + vector similarity  
      tag_relevance_score +         # Tag matching score
      pagerank_score +              # Document authority
      position_weight +             # Document position
      recency_factor                # Time-based boost
  )

  4. FULL SEARCH FLOW VỚI METADATA

  User Query: "How to implement machine learning models?"

  Step 1: Query Analysis
  ├── Extract query tags: {"machine_learning": 8, "implementation": 6, "models": 7}
  ├── Match with pre-generated questions: "How to build ML models?" 
  └── Prepare rank features: {TAG_FLD + ".machine_learning": 8, ...}

  Step 2: Multi-field Search (IN ELASTICSEARCH)
  ├── title_tks^10: "machine learning models"
  ├── question_tks^20: Match "How to build ML models?" (20x boost!)
  ├── important_kwd^30: Match "machine learning" keywords (30x boost!)
  ├── content_ltks^2: Standard content search
  └── rank_feature: Boost docs với matching tags (DURING SEARCH)

  Step 3: Vector Search
  ├── Dense retrieval: Embedding similarity
  ├── Fusion: 5% text + 95% vector
  └── Tag boosting: Documents với relevant tags get higher scores

  Step 4: Post-search Reranking
  ├── Tag relevance: Additional tag matching score
  ├── PageRank: Document authority boost
  ├── Position: In-document position weight
  └── Final ranking: Combined multi-signal score

  5. So sánh với LightRAG

  | Stage            | RAGFlow                                           | LightRAG
        |
  |------------------|---------------------------------------------------|---------------------------
  ------|
  | Query Processing | Extract tags từ query → Boost search fields       | Extract entities → Graph
  lookup |
  | During Search    | Multi-field search với metadata boosting          | Simple vector + entity
  matching |
  | Initial Ranking  | Hybrid fusion với tag/question/importance weights | Similarity-based ranking
        |
  | Post-search      | Multi-signal reranking với tags/PageRank/position | Basic similarity reranking
        |

  6. Kết luận

  RAGFlow metadata được sử dụng ở 3 stages:

  1. Pre-search: Query tag extraction và field preparation
  2. During search: Elasticsearch rank_feature boosting và multi-field search
  3. Post-search: Additional tag scoring và multi-signal reranking

  Điều này cho phép:
  - Documents với relevant tags được boost ngay trong search engine
  - Pre-generated questions match user intent với 20x boost
  - Important keywords get 30x boost trong search
  - Final results được fine-tuned với multiple relevance signals

  Khác biệt lớn với LightRAG: Metadata không chỉ dùng để rerank, mà được embedded deeply into search 
  process để achieve much higher precision ngay từ initial retrieval!

  Perfect! Bây giờ tôi có đầy đủ thông tin để phân tích chi tiết cách Elasticsearch lưu trữ metadata
  trong RAGFlow:

  Elasticsearch Storage Structure in RAGFlow

  1. Dynamic Mapping Templates

  RAGFlow sử dụng dynamic templates để tự động map field types dựa trên naming conventions:

  // mapping.json
  {
    "mappings": {
      "dynamic_templates": [
        // Text fields với tokenization
        {
          "tks": {
            "match": "*_tks",           // title_tks, question_tks, content_tks
            "mapping": {
              "type": "text",
              "similarity": "scripted_sim",  // Custom BM25-like scoring
              "analyzer": "whitespace",
              "store": true
            }
          }
        },
        {
          "ltks": {
            "match": "*_ltks",          // content_ltks, content_sm_ltks  
            "mapping": {
              "type": "text",
              "analyzer": "whitespace",
              "store": true
            }
          }
        },
        // Keywords cho exact matching
        {
          "kwd": {
            "match": "^(.*_(kwd|id|ids|uid|uids)|uid)$",  // important_kwd, question_kwd
            "mapping": {
              "type": "keyword",
              "similarity": "boolean",
              "store": true
            }
          }
        },
        // Rank features cho scoring
        {
          "rank_feature": {
            "match": "*_fea",           // pagerank_fea
            "mapping": {
              "type": "rank_feature"    // Elasticsearch native ranking
            }
          }
        },
        {
          "rank_features": {
            "match": "*_feas",          // tag_feas
            "mapping": {
              "type": "rank_features"   // Multiple ranking signals
            }
          }
        }
      ]
    }
  }

  2. Metadata Fields Storage

  Text Search Fields:

  # Mỗi document được lưu với các fields sau:

  "title_tks": "machine learning algorithms"           # Tokenized title
  "title_sm_tks": "machine learn algorithm"          # Fine-grained tokens

  "question_tks": "How to implement ML? What are algorithms?"  # Auto-generated questions tokenized
  "question_kwd": ["How to implement ML?", "What are algorithms?"]  # Raw questions

  "content_ltks": "This document describes machine learning..."   # Large tokens
  "content_sm_ltks": "This doc describe machine learn"          # Small tokens

  "important_kwd": ["machine learning", "algorithms", "neural networks"]  # Key terms
  "important_tks": "machine learning algorithms neural networks"          # Tokenized key terms

  Ranking & Metadata Fields:

  "pagerank_fea": 0.85,                    # Document authority score (rank_feature type)

  "tag_feas": {                            # Multiple tag relevance scores (rank_features type)
      "machine_learning": 9,
      "algorithms": 8,
      "neural_networks": 7,
      "deep_learning": 6
  },

  "doc_id": "doc_123",                     # Document identifier
  "kb_id": "kb_456",                       # Knowledge base identifier  
  "docnm_kwd": "ML_Guide.pdf",             # Document name
  "position_int": [1, 100, 200, 50, 150], # Page positions
  "page_num_int": [1, 2, 3],              # Page numbers

  Vector Fields:

  "q_768_vec": [0.1234, -0.5678, ...],    # 768-dim embedding (dense_vector type)
  "q_1536_vec": [0.2345, -0.6789, ...],   # 1536-dim embedding

  3. Multi-field Search Query Example

  Khi user search "machine learning algorithms", Elasticsearch query sẽ như thế này:

  {
    "query": {
      "bool": {
        "should": [
          // Multi-field text search với different boosts
          {
            "multi_match": {
              "query": "machine learning algorithms",
              "fields": [
                "title_tks^10",           // Title boost 10x
                "title_sm_tks^5",         // Title fine-grained 5x  
                "important_kwd^30",       // Important keywords 30x boost!
                "important_tks^20",       // Important tokens 20x boost!
                "question_tks^20",        // Questions 20x boost!
                "content_ltks^2",         // Content 2x
                "content_sm_ltks^1"       // Content fine-grained 1x
              ]
            }
          },
          // Vector similarity search
          {
            "knn": {
              "field": "q_768_vec",
              "query_vector": [0.1234, -0.5678, ...],
              "k": 1000,
              "num_candidates": 2000
            }
          },
          // Rank feature boosting với extracted tags
          {
            "rank_feature": {
              "field": "tag_feas.machine_learning",
              "boost": 8,               // Tag relevance score từ query
              "linear": {}
            }
          },
          {
            "rank_feature": {
              "field": "tag_feas.algorithms",
              "boost": 7,
              "linear": {}
            }
          },
          // PageRank boost
          {
            "rank_feature": {
              "field": "pagerank_fea",
              "boost": 10,
              "linear": {}
            }
          }
        ]
      }
    }
  }

  4. Field Population Process

  During Document Indexing:

  # task_executor.py - When processing documents

  # 1. Basic tokenization
  d["title_tks"] = rag_tokenizer.tokenize(title)
  d["content_ltks"] = rag_tokenizer.tokenize(content)
  d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(content)

  # 2. Auto-generate questions về content
  questions = question_proposal(chat_mdl, content, topn=3)
  d["question_kwd"] = questions.split("\n")
  d["question_tks"] = rag_tokenizer.tokenize("\n".join(questions))

  # 3. Extract important keywords
  important_terms = extract_key_terms(content)  # From LLM or NLP
  d["important_kwd"] = important_terms
  d["important_tks"] = rag_tokenizer.tokenize(" ".join(important_terms))

  # 4. Auto-tag với LLM
  tags = content_tagging(chat_mdl, content, all_tags, examples)
  d["tag_feas"] = tags  # {"machine_learning": 9, "algorithms": 8, ...}

  # 5. Set PageRank
  d["pagerank_fea"] = calculate_document_authority(doc)

  5. Search Performance Impact

  Field Boosting trong Elasticsearch:

  - important_kwd^30: Documents có exact keyword matches get massive boost
  - question_tks^20: Pre-generated questions match user intent với high relevance
  - title_tks^10: Titles get significant boost
  - content_ltks^2: Standard content search

  Rank Features:

  - tag_feas: Dynamic boosting based on query-extracted tags
  - pagerank_fea: Authority-based boosting
  - Linear combination: Elasticsearch combines all signals into final score

  6. Storage Optimization

  // Elasticsearch optimizations in mapping.json

  {
    "similarity": {
      "scripted_sim": {
        "type": "scripted",
        "script": {
          // Custom BM25-like formula cho better relevance
          "source": "double idf = Math.log(1+(field.docCount-term.docFreq+0.5)/(term.docFreq + 
  0.5))/Math.log(1+((field.docCount-0.5)/1.5)); return query.boost * idf * Math.min(doc.freq, 1);"
        }
      }
    }
  }

  7. So sánh với LightRAG

  | Aspect      | RAGFlow ES Storage                     | LightRAG                       |
  |-------------|----------------------------------------|--------------------------------|
  | Field Types | 15+ specialized fields                 | Simple text + embeddings       |
  | Boosting    | Multi-level field boosting (1x to 30x) | Basic similarity               |
  | Metadata    | Rich metadata với rank_features        | Basic entity storage           |
  | Query Types | Multi-match + kNN + rank_feature       | Vector similarity + basic text |
  | Ranking     | Native ES ranking + custom similarity  | Post-processing reranking      |

  Kết luận:

  RAGFlow leverage Elasticsearch's advanced features để create một "semantic search engine":
  - Dynamic templates: Auto-map fields based on naming
  - Multi-field search: Different content types với appropriate boosts
  - Rank features: Native Elasticsearch boosting cho tags/PageRank
  - Custom similarity: Optimized scoring functions
  - Rich metadata: 15+ fields per document cho comprehensive search

  Điều này cho phép achieve enterprise-grade search quality với low latency và high relevance!

⏺ Chính xác! Bạn hiểu hoàn toàn đúng. Đây là cách RAGFlow hoạt động:

  RAGFlow 2-Layer Reranking Architecture

  1. Document = Chunk Relationship

  # Mỗi document trong Elasticsearch = 1 chunk
  {
    "_id": "chunk_12345",
    "_source": {
      "doc_id": "doc_789",           # Original document ID
      "kb_id": "kb_456",
      "content_ltks": "Machine learning is...",  # This chunk's content
      "title_tks": "ML Chapter 3",   # Chunk title
      "page_num_int": [15],          # Page 15
      "position_int": [15, 100, 400, 50, 150],  # Position on page
      "important_kwd": ["machine learning", "algorithms"],
      "question_tks": "What is machine learning?",
      "tag_feas": {"ML": 9, "algorithms": 8},
      "pagerank_fea": 0.85,
      "q_768_vec": [0.123, -0.456, ...]
    }
  }

  2. Layer 1: Elasticsearch Native Reranking

  TRONG Elasticsearch Query:

  # search.py:117-118
  res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy,
                             offset, limit, idx_names, kb_ids,
                             rank_feature=rank_feature)  # ← Native ES reranking

  Elasticsearch tự động combine scores:

  {
    "query": {
      "bool": {
        "should": [
          // Text search với field boosting
          {"multi_match": {"query": "ML", "fields": ["important_kwd^30", "question_tks^20"]}},

          // Vector search
          {"knn": {"field": "q_768_vec", "query_vector": [...]}},

          // Rank features boosting
          {"rank_feature": {"field": "tag_feas.machine_learning", "boost": 8}},
          {"rank_feature": {"field": "pagerank_fea", "boost": 10}}
        ]
      }
    }
  }

  // Elasticsearch returns sorted results với combined score

  3. Layer 2: Post-ES Model-based Reranking

  Method 1: Internal Reranking (không có external model):

  # search.py:279-316
  def rerank(self, sres, query, tkweight=0.3, vtweight=0.7):
      # Recalculate similarities với extracted features
      for i in sres.ids:
          content_ltks = sres.field[i]["content_ltks"].split()
          title_tks = sres.field[i].get("title_tks", "").split()
          question_tks = sres.field[i].get("question_tks", "").split()
          important_kwd = sres.field[i].get("important_kwd", [])

          # Weighted combination: content + title*2 + important*5 + question*6
          tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6

      # Calculate hybrid similarity
      sim = self.qryr.hybrid_similarity(query_vector, chunk_embeddings, keywords, tks, tkweight,
  vtweight)

      # Add rank feature scores
      rank_fea = self._rank_feature_scores(rank_feature, sres)

      return sim + rank_fea  # Final reranked scores

  Method 2: Model-based Reranking:

  # search.py:318-339
  def rerank_by_model(self, rerank_mdl, sres, query):
      # Token similarity (internal)
      tksim = self.qryr.token_similarity(keywords, ins_tw)

      # Model-based similarity (external reranker như BGE)
      vtsim, _ = rerank_mdl.similarity(query, [" ".join(tks) for tks in ins_tw])

      # Rank feature scores
      rank_fea = self._rank_feature_scores(rank_feature, sres)

      # Weighted combination: tkweight * (token_sim + rank_fea) + vtweight * model_sim
      return tkweight * (np.array(tksim) + rank_fea) + vtweight * vtsim

  4. Complete Search Flow

  User Query: "How to implement machine learning?"

  ┌─ Layer 1: Elasticsearch Native Reranking ─┐
  │                                             │
  │ 1. Multi-field search:                     │
  │    ├── important_kwd^30: "machine learning"│
  │    ├── question_tks^20: "How to implement?"│
  │    ├── title_tks^10: content titles        │
  │    └── content_ltks^2: main content        │
  │                                             │
  │ 2. Vector similarity: q_768_vec            │
  │                                             │
  │ 3. Rank features:                          │
  │    ├── tag_feas.machine_learning: boost 8  │
  │    └── pagerank_fea: boost 10              │
  │                                             │
  │ 4. Fusion: 5% text + 95% vector            │
  │                                             │
  │ → Returns top 1000 pre-ranked chunks       │
  └─────────────────────────────────────────────┘
                          ↓
  ┌─ Layer 2: Post-ES Model Reranking ─────────┐
  │                                             │
  │ 5. Take top 64 chunks từ ES results        │
  │                                             │
  │ 6a. Internal reranking:                    │
  │     ├── Token similarity                   │
  │     ├── Weighted features (question*6)     │
  │     └── Rank feature scores                │
  │                                             │
  │ 6b. Model reranking (if available):        │
  │     ├── BGE reranker similarity            │
  │     ├── Combined với token similarity      │
  │     └── Weighted: 0.3*token + 0.7*model    │
  │                                             │
  │ → Returns final top K results              │
  └─────────────────────────────────────────────┘

  5. Why 2 Layers?

  Layer 1 (ES Native) Advantages:

  - Fast: Native Elasticsearch scoring
  - Scalable: Can handle millions of chunks
  - Multi-signal: Combines text + vector + metadata trong 1 query
  - Field boosting: Different weights cho different content types

  Layer 2 (Model Reranking) Advantages:

  - High precision: Advanced reranking models (BGE, etc.)
  - Context-aware: Better understanding of query-document relevance
  - Limited scope: Only top 64 candidates (RERANK_LIMIT = 64)
  - Expensive: Too costly để run trên all results

  6. Configuration

  # search.py:347-377
  def retrieval(..., rerank_mdl=None, ...):
      RERANK_LIMIT = 64  # Only rerank top 64 từ ES

      # Get initial results từ ES
      sres = self.search(req, index_names, kb_ids, emb_mdl, rank_feature=rank_feature)

      if rerank_mdl and sres.total > 0:
          # Use external reranking model
          sim = self.rerank_by_model(rerank_mdl, sres, question)
      else:
          # Use internal reranking
          sim = self.rerank(sres, question, rank_feature=rank_feature)

  Kết luận:

  - Mỗi doc trong ES = 1 chunk với rich metadata
  - ES native reranking handles bulk filtering (1M+ chunks → 1K chunks)
  - Model reranking handles precision optimization (1K chunks → 64 final results)
  - 2-layer approach balances performance vs accuracy

  Đây chính là lý do RAGFlow có thể achieve both high scalability và high precision trong enterprise
  environments!  

⏺ Chính xác 100%! Boost càng cao thì chunk càng có khả năng lên top, và chúng ta hoàn toàn có thể 
  control các con số này để tối ưu search quality.

  RAGFlow Boost Configuration & Control

  1. Default Boost Values

  Field Boost (cố định trong query fields):

  # rag/nlp/query.py:30-38
  self.query_fields = [
      "title_tks^10",           # Title: 10x boost
      "title_sm_tks^5",         # Title fine-grained: 5x boost
      "important_kwd^30",       # Important keywords: 30x boost 🔥
      "important_tks^20",       # Important tokens: 20x boost
      "question_tks^20",        # Questions: 20x boost
      "content_ltks^2",         # Content: 2x boost
      "content_sm_ltks^1",      # Content fine-grained: 1x (base)
  ]

  Rank Feature Boost (dynamic):

  # search.py:350 - DEFAULT values
  rank_feature: dict = {PAGERANK_FLD: 10}  # PageRank: 10x boost

  # tag.py:499-501 - Tag relevance scores được calculate dynamic
  tag_fea = sorted([(tag, score) for tag, score in aggs], key=lambda x: x[1] * -1)[:topn_tags]
  # Top tags get scores from 1-10, được dùng làm boost values

  2. Dynamic Tag Boost Calculation

  # search.py:488-501 - Tag query analysis
  def tag_query(self, question, tenant_ids, kb_ids, all_tags, topn_tags=3, S=1000):
      # Calculate tag relevance scores từ user query
      tag_fea = sorted([
          (tag, round(0.1 * (count + 1) / (total_count + S) / max(1e-6, all_tags.get(tag, 0.0001))))
          for tag, count in aggregations
      ], key=lambda x: x[1] * -1)[:topn_tags]

      return {tag.replace(".", "_"): max(1, score) for tag, score in tag_fea}

  # Example output:
  # {"machine_learning": 8, "algorithms": 7, "neural_networks": 6}

  3. Configuration Points để Control Boost

  A. Knowledge Base Level:

  # topn_tags: Control how many tags to extract and their boost
  kb.parser_config = {
      "topn_tags": 3,           # Top 3 tags từ mỗi query (default)
      "tag_kb_ids": [kb1, kb2]  # Which KBs to extract tags from
  }

  # Có thể set từ 1-10 tags (validation_utils.py:352)
  topn_tags: Annotated[int, Field(default=1, ge=1, le=10)]

  B. Query Level:

  # search.py:347 - Có thể override rank_feature khi call retrieval
  def retrieval(..., rank_feature: dict = {PAGERANK_FLD: 10}):

  # Example custom boost:
  custom_boost = {
      PAGERANK_FLD: 15,           # Increase PageRank boost to 15x
      "machine_learning": 12,     # Boost ML-related chunks 12x
      "algorithms": 10,           # Boost algorithm chunks 10x
      "tutorials": 8              # Boost tutorial chunks 8x
  }

  C. Elasticsearch Level:

  # es_conn.py:215-219 - Actual boost application
  if bqry and rank_feature:
      for fld, sc in rank_feature.items():
          if fld != PAGERANK_FLD:
              fld = f"{TAG_FLD}.{fld}"  # Convert to tag_feas.machine_learning
          bqry.should.append(Q("rank_feature", field=fld, linear={}, boost=sc))

  4. Real-world Boost Impact

  Example Query: "How to implement neural networks?"

  # Step 1: Extract tags từ query
  query_tags = label_question("How to implement neural networks?", kbs)
  # → {"neural_networks": 9, "implementation": 7, "tutorials": 5}

  # Step 2: Elasticsearch query với multiple boost levels
  {
    "query": {
      "bool": {
        "should": [
          // Field boosts (fixed)
          {"multi_match": {
            "query": "implement neural networks",
            "fields": [
              "important_kwd^30",      # 30x nếu match exact keywords
              "question_tks^20",       # 20x nếu match pre-generated questions
              "title_tks^10"           # 10x nếu match in title
            ]
          }},

          // Vector similarity
          {"knn": {"field": "q_768_vec", "query_vector": [...]}},

          // Dynamic tag boosts
          {"rank_feature": {"field": "tag_feas.neural_networks", "boost": 9}},  // 9x boost!
          {"rank_feature": {"field": "tag_feas.implementation", "boost": 7}},   // 7x boost!
          {"rank_feature": {"field": "tag_feas.tutorials", "boost": 5}},        // 5x boost!

          // PageRank boost
          {"rank_feature": {"field": "pagerank_fea", "boost": 10}}              // 10x boost!
        ]
      }
    }
  }

  5. Practical Tuning Strategies

  A. Domain-specific Boosting:

  # For technical documentation:
  tech_boost = {
      PAGERANK_FLD: 12,
      "tutorials": 15,        # Boost tutorials higher
      "examples": 12,         # Boost code examples
      "api_docs": 10,         # Boost API documentation
      "troubleshooting": 8    # Boost troubleshooting content
  }

  # For customer support:
  support_boost = {
      PAGERANK_FLD: 10,
      "faq": 20,             # FAQs get highest boost
      "solutions": 15,       # Solution-oriented content
      "common_issues": 12,   # Common problems
      "step_by_step": 10     # Step-by-step guides
  }

  B. User-specific Boosting:

  # For beginners:
  beginner_boost = {
      "basics": 15,
      "introduction": 12,
      "getting_started": 10,
      "advanced": 2          # Lower boost for advanced content
  }

  # For experts:
  expert_boost = {
      "advanced": 15,
      "technical": 12,
      "implementation": 10,
      "basics": 2            # Lower boost for basic content
  }

  C. Time-sensitive Boosting:

  # Recent content boost
  recent_boost = {
      PAGERANK_FLD: 8,       # Lower historical importance
      "latest": 20,          # Boost latest information
      "updated": 15,         # Recently updated content
      "news": 12             # News and announcements
  }

  6. Boost Impact on Search Results

  Without Custom Boosting:

  Query: "neural network implementation"
  Results:
  1. General ML paper (score: 0.85)
  2. Algorithm overview (score: 0.82)
  3. Neural network tutorial (score: 0.79) ← Target content lower!

  With Custom Boosting:

  Query: "neural network implementation"
  + boost: {"neural_networks": 12, "tutorials": 10, "implementation": 8}

  Results:
  1. Neural network tutorial (score: 1.95) ← Target content on top! 
  2. NN implementation guide (score: 1.87)
  3. General ML paper (score: 0.85)

  7. Configuration Best Practices

  1. Start with defaults: topn_tags=3, PAGERANK_FLD=10
  2. Monitor search quality: Track click-through rates, user satisfaction
  3. A/B test boost values: Compare different boost configurations
  4. Domain-specific tuning: Different boost strategies for different content types
  5. User feedback integration: Adjust boosts based on user behavior

  Kết luận:

  RAGFlow cung cấp multiple layers of boost control:
  - Field boosts: 1x to 30x cho different content fields
  - Tag boosts: 1-10 dynamic scores cho query-relevant tags
  - PageRank boosts: 10x default cho document authority
  - Custom overrides: Có thể set bất kỳ boost value nào

  Điều này cho phép fine-tune search behavior để optimize cho specific domains, user types, và
  content strategies!  

⏺ Perfect! Bây giờ tôi thấy rõ cách RAGFlow làm. Có 2 phương pháp song song:

  RAGFlow Tag Scoring: Statistical + LLM

  1. Lúc INDEXING (Document → Tags)

  Method 1: Statistical Analysis (Fast)

  # search.py:475-486 - tag_content()
  def tag_content(self, tenant_id, kb_ids, doc, all_tags, topn_tags=3):
      # Search similar content trong existing index
      match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"])
      res = self.dataStore.search([...], [match_txt], [...], ["tag_kwd"])

      # Aggregate tags từ similar documents
      aggs = self.dataStore.getAggregation(res, "tag_kwd")
      # → [("machine_learning", 45), ("algorithms", 30), ("neural_networks", 20), ...]

      # Calculate TF-IDF like score với Bayesian smoothing
      tag_fea = sorted([
          (tag, round(0.1 * (count + 1) / (total_count + S) / max(1e-6, all_tags.get(tag, 0.0001))))
          for tag, count in aggs
      ], key=lambda x: x[1] * -1)[:topn_tags]

      doc[TAG_FLD] = {tag: score for tag, score in tag_fea if score > 0}
      return True  # If successful

  Logic: Nếu document tương tự có tags nào thì document này có thể có tags đó.

  Method 2: LLM Analysis (Accurate but Slow)

  # task_executor.py:402-424 - Nếu statistical method không work
  if not settings.retrievaler.tag_content(...):  # Statistical failed
      # Use LLM với few-shot learning
      picked_examples = random.choices(examples, k=2)  # Get 2 examples
      cached = content_tagging(chat_mdl, content, all_tags, picked_examples, topn=topn_tags)
      d[TAG_FLD] = cached  # {"machine_learning": 9, "tutorials": 7, ...}

  LLM Prompt Example:

  ## Role: You are a text analyzer.

  ## Task: Add tags based on examples and tag set.

  # TAG SET
  machine_learning, neural_networks, algorithms, tutorials, implementation, optimization, ...

  # Example 1
  ### Text Content
  This tutorial explains how to build neural networks using Python and TensorFlow...

  Output:
  {"neural_networks": 9, "tutorials": 8, "python": 7}

  # Example 2  
  ### Text Content
  Advanced optimization techniques for deep learning models including Adam optimizer...

  Output:
  {"optimization": 9, "deep_learning": 8, "algorithms": 6}

  # Real Data
  ### Text Content
  Implementing convolutional neural networks for image classification tasks...

  Output: {"neural_networks": 9, "implementation": 8, "image_processing": 6}

  2. Lúc SEARCH (Query → Tag Relevance)

  Statistical Query Analysis:

  # search.py:488-501 - tag_query()
  def tag_query(self, question, tenant_ids, kb_ids, all_tags, topn_tags=3):
      # Search documents matching user query
      match_txt, _ = self.qryr.question(question, min_match=0.0)
      res = self.dataStore.search([...], [match_txt], [...], ["tag_kwd"])

      # Aggregate tags từ matching documents
      aggs = self.dataStore.getAggregation(res, "tag_kwd")
      # → [("neural_networks", 80), ("implementation", 60), ("tutorials", 40)]

      # Calculate relative importance với IDF-like formula
      total_count = np.sum([count for _, count in aggs])
      tag_scores = []
      for tag, count in aggs:
          # Formula: (count + 1) / (total + smoothing) / global_frequency
          score = round(0.1 * (count + 1) / (total_count + S) / max(1e-6, all_tags.get(tag, 0.0001)))
          tag_scores.append((tag, score))

      # Return top scoring tags
      return {tag: max(1, score) for tag, score in sorted(tag_scores, reverse=True)[:topn_tags]}

  3. Tag Scoring Logic Chi Tiết

  Formula Breakdown:

  # search.py:499
  score = 0.1 * (count + 1) / (total_count + S) / max(1e-6, all_tags.get(tag, 0.0001))

  # Where:
  # count: Số lần tag xuất hiện trong matching docs
  # total_count: Tổng số tags trong matching docs  
  # S: Smoothing parameter (1000)
  # all_tags.get(tag): Global frequency của tag trong toàn bộ KB

  Why This Formula Works:

  1. (count + 1) / (total_count + S): TF (term frequency) với Bayesian smoothing
  2. / all_tags.get(tag): IDF (inverse document frequency) - rare tags get higher scores
  3. 0.1: Scaling factor để normalize về range 1-10
  4. max(1, score): Minimum score = 1

  Example Calculation:

  # Query: "neural network implementation"
  # Matching docs có tags: [("neural_networks", 50), ("implementation", 30), ("tutorials", 20)]
  # total_count = 100, S = 1000

  # all_tags (global frequencies):
  all_tags = {
      "neural_networks": 0.05,    # Rare tag (5% of docs)
      "implementation": 0.15,     # Medium tag (15% of docs)
      "tutorials": 0.25          # Common tag (25% of docs)
  }

  # Scores:
  neural_networks_score = 0.1 * (50 + 1) / (100 + 1000) / 0.05 = 0.927 → 1
  implementation_score = 0.1 * (30 + 1) / (100 + 1000) / 0.15 = 0.188 → 1
  tutorials_score = 0.1 * (20 + 1) / (100 + 1000) / 0.25 = 0.076 → 1

  # But với proper scaling → {"neural_networks": 9, "implementation": 6, "tutorials": 3}

  4. Tag Source (all_tags) Generation

  Where All Tags Come From:

  # search.py:470-473 - all_tags_in_portion()
  def all_tags_in_portion(self, tenant_id, kb_ids):
      # Get all existing tags từ Elasticsearch
      res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids,
   ["tag_kwd"])
      res = self.dataStore.getAggregation(res, "tag_kwd")
      # → [("machine_learning", 500), ("algorithms", 300), ...]

      total = np.sum([count for _, count in res])
      return {tag: (count + 1) / (total + S) for tag, count in res}  # Global frequencies

  Tag Evolution:

  1. Initially: Empty tag vocabulary
  2. Manual tags: Admin adds some seed tags
  3. LLM generates more: Based on content analysis
  4. Statistical propagation: Similar docs get similar tags
  5. Continuous learning: Tag vocabulary grows over time

  5. Why This Approach Works

  Statistical Benefits:

  - Fast: No LLM call needed for every search
  - Context-aware: Tags based on actual query-document matches
  - Scalable: Works với millions of documents

  LLM Benefits:

  - Semantic understanding: Understands content meaning
  - Consistent: Few-shot examples ensure consistency
  - Comprehensive: Can generate tags for any content

  Combined Power:

  - Statistical for search speed (sub-second)
  - LLM for indexing accuracy (can take time)
  - Best of both worlds: Fast + accurate

  Kết luận:

  RAGFlow tag scoring không chỉ dựa vào LLM mà combine:
  1. Statistical analysis: TF-IDF like scoring từ similar documents
  2. LLM few-shot learning: Semantic understanding với examples
  3. Global frequency: Rare tags get higher importance
  4. Bayesian smoothing: Handle sparse data gracefully

  Điều này tạo ra tag scores rất contextual và accurate cho both indexing và search!  


⏺ Tìm thấy! RAGFlow có 2 functions khác nhau cho reranking:

  1. Function rerank() (search.py:316)

  return sim + rank_fea, tksim, vtsim
  - rank_fea KHÔNG phụ thuộc tkweight
  - Chỉ cộng thêm vào hybrid similarity

  2. Function rerank_by_model() (search.py:339)

  return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim
  - rank_fea CÓ phụ thuộc tkweight
  - rank_fea được nhân với tkweight

  Công thức chi tiết:

  Method 1 (rerank):

  final_score = (vector_sim × vtweight + term_sim × tkweight) + rank_fea
  final_score = hybrid_similarity + rank_fea

  Method 2 (rerank_by_model):

  final_score = tkweight × (term_sim + rank_fea) + vtweight × vector_sim
  final_score = tkweight × term_sim + tkweight × rank_fea + vtweight × vector_sim

  Trong trường hợp của bạn:

  Nếu dùng Method 2 với tkweight=1.0, vtweight=0.0:
  final_score = 1.0 × (0.00 + rank_fea) + 0.0 × 0.20
  final_score = 1.0 × rank_fea + 0
  final_score = rank_fea = 344.66

  rank_fea calculation:

  def _rank_feature_scores():
      # Tag similarity + PageRank  
      return np.array(tag_similarity) * 10.0 + pagerank_scores

  Kết luận: Score 344.66 = rank_fea × tkweight (nếu dùng rerank_by_model) hoặc chỉ là rank_fea (nếu dùng rerank).  