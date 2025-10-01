curl -X GET "http://admin:infini_rag_flow_OS_01@localhost:1201/_cat/indices?v"

curl -X GET "localhost:1201/ragflow_87fec81c92ab11f091fc85c4f939180f*/_search" \
    -u "admin:infini_rag_flow_OS_01" \
    -H 'Content-Type: application/json' \
    -k \
    -d'{
      "query": {
        "bool": {
          "must": [
            {"term": {"kb_id": "2aaa99dc93c511f0a2b7db8ffb6ddb03"}},
           {"term": {"knowledge_graph_kwd": "community_report"}}
          ]
        }
      },
      "_source": {
        "excludes": ["*_vec", "vector"]
      },
      "size": 5
    }'

curl -X GET "localhost:1201/ragflow_87fec81c92ab11f091fc85c4f939180f/_search" -H 'Content-Type: application/json' -H 'Authorization: Basic YWRtaW46aW5maW5pX3JhZ19mbG93X09TXzAx' \
 -d'
  {
    "query": {
      {"kb_id":"af3d6bfa92c811f097fcdf00eef63319"}
    }
  }'

af3d6bfa92c811f097fcdf00eef63319
af3d6bfa92c811f097fcdf00eef63319
9dd9386892b811f097fcdf00eef63319
  curl -X GET "localhost:1201/ragflow_87fec81c92ab11f091fc85c4f939180f*/_search" \
    -u "admin:infini_rag_flow_OS_01" \
    -H 'Content-Type: application/json' \
    -k \
    -d'{
      "query": {
        "bool": {
          "must": [
            {"term": {"kb_id": "ffd857d894f711f0a54dfbbcaf2d13ed"}},
            {"term": {"doc_id": "0bee0b1c94f811f0a54dfbbcaf2d13ed"}}
          ]
        }
      },
      "_source": {
        "excludes": ["*_vec", "vector"]
      },
      "size": 5
    }'


curl -X GET "localhost:1201/ragflow_87fec81c92ab11f091fc85c4f939180f*/_search" \
    -u "admin:infini_rag_flow_OS_01" \
    -H 'Content-Type: application/json' \
    -k \
    -d'{
      "query": {
        "bool": {
          "must": [
            {"term": {"kb_id": "af3d6bfa92c811f097fcdf00eef63319"}},
            {"term": {"_id": "06a495ccc467c46f"}}
          ]
        }
      },
      "_source": {
        "excludes": ["*_vec", "vector"]
      },
      "size": 5
    }'

Perfect! Giờ tôi hiểu rõ cách RAGFlow lưu graph documents khác với regular documents:

  RAGFlow Graph DB Document Types:

  1. Document Classification trong ES:

  // Regular documents (thường)
  {
    "knowledge_graph_kwd": null,  // Hoặc không có field này
    "content_ltks": "thẻ tín dụng sacombank",
    "content_with_weight": "Thẻ tín dụng Sacombank...",
    "doc_id": "doc_123",
    "kb_id": "kb_456"
  }

  // Graph documents (khác)
  {
    "knowledge_graph_kwd": "entity|relation|subgraph|graph"  // KEY DIFFERENCE!
  }

  2. Các loại Graph Documents:

  A. Entity Documents:

  {
    "knowledge_graph_kwd": "entity",           // Type identifier
    "entity_kwd": "Neural Networks",           // Entity name
    "entity_type_kwd": "technology",           // Entity type
    "content_with_weight": "Neural networks are...", // Entity description
    "rank_flt": 0.85,                         // PageRank score
    "n_hop_with_weight": [                    // Pre-computed graph paths
      {
        "path": ["Neural Networks", "ML", "AI"],
        "weights": [0.8, 0.7, 0.6]
      }
    ]
  }

  B. Relationship Documents:

  {
    "knowledge_graph_kwd": "relation",        // Type identifier
    "from_entity_kwd": "Neural Networks",     // Source entity
    "to_entity_kwd": "Deep Learning",         // Target entity  
    "content_with_weight": "Neural networks are foundation of deep learning...",
    "weight_int": 8                           // Relationship strength
  }

  C. Subgraph Documents:

  {
    "knowledge_graph_kwd": "subgraph",        // Community/cluster
    "entities_list": ["A", "B", "C"],         // Entities trong community
    "community_id": "comm_123",
    "content_with_weight": "This cluster represents..."
  }

  3. Key Differences:

  Regular Documents:

  - Purpose: Lưu text content để search
  - Fields: content_ltks, title_tks, doc_id
  - Search: Text + vector similarity
  - Structure: Flat document structure

  Graph Documents:

  - Purpose: Lưu graph structure và relationships
  - Fields: entity_kwd, from_entity_kwd, to_entity_kwd, rank_flt
  - Search: Multi-dimensional (keywords, types, relationships, n-hop)
  - Structure: Interconnected entity-relationship structure

  4. Search Strategy Differences:

  Regular Search:

  # Simple content search
  filters = {"kb_id": kb_ids}
  matchText = self.qryr.question(question)

  Graph Search:

  # Entity search
  filters["knowledge_graph_kwd"] = "entity"
  matchDense = self.get_vector(", ".join(keywords), emb_mdl)

  # Relationship search  
  filters["knowledge_graph_kwd"] = "relation"
  fields = ["from_entity_kwd", "to_entity_kwd", "weight_int"]

  5. Storage Benefits:

  Traditional Graph DB (Neo4j):
  - Graph traversal tốn memory
  - Complex query language (Cypher)
  - Separate infrastructure

  RAGFlow's ES Graph Storage:
  - Leverage ES indexing power
  - Vector + text + graph trong cùng storage
  - Pre-computed n-hop paths → fast retrieval
  - Horizontal scaling với ES clusters

  Kết luận: RAGFlow dùng knowledge_graph_kwd như discriminator field để phân biệt graph docs với regular docs, cho phép unified 
  storage nhưng specialized search cho từng loại!    

  Perfect! Giờ tôi có đủ thông tin để phân tích cách RAGFlow lưu communities trong ES:

  RAGFlow Communities Storage trong Elasticsearch:

  1. Community Processing Flow:

  A. Subgraph Creation (per document):

  # graphrag/general/index.py:174-184
  chunk = {
      "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False),
      "knowledge_graph_kwd": "subgraph",           # Individual document subgraph
      "kb_id": kb_id,
      "source_id": [doc_id],                       # Source document
      "available_int": 0,
      "removed_kwd": "N",
  }

  B. Community Detection & Reports:

  # graphrag/general/index.py:269-289  
  chunk = {
      "id": get_uuid(),
      "docnm_kwd": stru["title"],                  # Community title
      "title_tks": rag_tokenizer.tokenize(stru["title"]),
      "content_with_weight": json.dumps({          # Community report + evidences
          "report": rep,
          "evidences": "\n".join([f.get("explanation", "") for f in stru["findings"]])
      }, ensure_ascii=False),
      "content_ltks": rag_tokenizer.tokenize(obj["report"] + " " + obj["evidences"]),
      "knowledge_graph_kwd": "community_report",   # KEY: Community type
      "weight_flt": stru["weight"],                # Community importance score
      "entities_kwd": stru["entities"],            # List of entities in community
      "important_kwd": stru["entities"],           # Same as entities for boost
      "kb_id": kb_id,
      "source_id": list(doc_ids),                  # Multiple source documents
      "available_int": 0,
  }

  2. Community Document Structure trong ES:

  {
    "_index": "ragflow_tenant123",
    "_source": {
      "knowledge_graph_kwd": "community_report",   // Identifies as community
      "docnm_kwd": "Financial Services Community",
      "title_tks": "financial services community",
      "content_with_weight": "{\"report\": \"This community represents financial institutions and their services...\", 
  \"evidences\": \"Evidence 1: Banks offer credit cards...\\nEvidence 2: Payment processing...\"}",
      "content_ltks": "this community represents financial institutions credit cards payment processing...",
      "weight_flt": 0.85,                         // Community importance
      "entities_kwd": [                           // Entities in this community
        "Sacombank",
        "VISA",
        "Credit Cards",
        "Payment System"
      ],
      "important_kwd": [                          // Same for search boosting
        "Sacombank",
        "VISA",
        "Credit Cards"
      ],
      "kb_id": "kb_123",
      "source_id": ["doc_1", "doc_2", "doc_3"],  // Source documents
      "available_int": 0,
      "q_768_vec": [0.1, -0.2, 0.8, ...]        // Embedding của community content
    }
  }

  3. Community Search & Retrieval:

  # Search for community reports
  filters = {
      "knowledge_graph_kwd": "community_report",  // Only community documents
      "kb_id": kb_ids
  }

  # Optional: Filter by specific entities
  filters["entities_kwd"] = ["Sacombank", "VISA"]

  es_res = self.dataStore.search(
      ["content_with_weight", "entities_kwd", "weight_flt"],
      [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids
  )

  4. Key Features của Community Storage:

  A. Hierarchical Structure:

  Documents → Subgraphs → Merged Graph → Communities → Community Reports

  B. Multi-level Indexing:

  - Individual subgraphs: knowledge_graph_kwd: "subgraph"
  - Community reports: knowledge_graph_kwd: "community_report"
  - Entities: knowledge_graph_kwd: "entity"
  - Relations: knowledge_graph_kwd: "relation"

  C. Rich Metadata:

  - weight_flt: Community importance score
  - entities_kwd: All entities in community (array)
  - source_id: Multiple source documents (array)
  - content_with_weight: Structured JSON với report + evidences

  5. Community vs Regular Documents:

  | Aspect     | Regular Documents         | Community Documents                         |
  |------------|---------------------------|---------------------------------------------|
  | Type Field | knowledge_graph_kwd: null | knowledge_graph_kwd: "community_report"     |
  | Content    | Raw text content          | LLM-generated community summary + evidences |
  | Entities   | Extracted entities        | Clustered community entities                |
  | Source     | Single document           | Multiple source documents                   |
  | Purpose    | Content search            | Graph reasoning & community insights        |

  6. Search Benefits:

  1. Community-aware Search: Filter by knowledge_graph_kwd: "community_report"
  2. Entity-based Filtering: Search communities containing specific entities
  3. Hierarchical Context: Access both individual documents và community summaries
  4. Importance Ranking: Use weight_flt để rank communities
  5. Evidence Tracing: source_id array để trace back tới original documents

  Kết luận: RAGFlow communities được lưu như special documents trong ES với rich metadata, cho phép graph-aware search và
  community-based reasoning!


  {"took":21,"timed_out":false,"_shards":{"total":2,"successful":2,"skipped":0,"failed":0},"hits":{"total":{"value":1,"relation":"eq"},"max_score":2.0,"hits":[{"_index":"ragflow_87fec81c92ab11f091fc85c4f939180f","_id":"79f1d4ed377e6a9f","_score":2.0,"_source":{"question_tks":"phí thường niên của thẻ sacombank platinum american express là bao nhiêu và khi nào chủ thẻ được miễn phí phí chuyển đổi ngoại tệ áp dụng cho giao dịch bằng thẻ này là bao nhiêu phần trăm hạn mức tín dụng tối thiểu và tối đa của thẻ sacombank platinum american express là bao nhiêu","content_with_weight":"Thẻ Sacombank Platinum American Express\n\n# Biểu phí\r\n\r\n**1. Phí phát hành:** Miễn phí.\r\n\r\n**2. Phí thường niên:**\r\n\r\nThẻ chính: 499.000 VND. Thẻ phụ: Miễn phí.\r\n\r\nTừ năm thứ 2 trở đi, chủ thẻ tín dụng được miễn phí thường niên khi đạt doanh số giao dịch. Chi tiết [tại đây](https://www.sacombank.com.vn/ca-nhan/the/dich-vu-the/mien-giam-phi-thuong-nien.html).  \r\n\r\n**3. Phí chuyển đổi ngoại tệ:** 1,99% số tiền giao dịch.\r\n\r\n**4\\. Phí khác**: Tham khảo biểu phí [tại đây](https://www.sacombank.com.vn/cong-cu/bieu-phi.html#khcn).\r\n\r\n# Hạn mức tín dụng\r\n\r\nTừ 40 triệu VND trở lên, tối đa không giới hạn.","create_time":"2025-09-20 16:26:58","tag_feas":{"Sacombank Platinum American Express":10,"Phí và Hạn mức":9,"Tổng quan":6},"create_timestamp_flt":1.758360418243993E9,"content_ltks":"thẻ sacombank platinum american express biểu phí 1 phí phát hành miễn phí 2 phí thường niên thẻ chính 499 000 vnd thẻ phụ miễn phí từ năm thứ 2 trở đi chủ thẻ tín dụng được miễn phí thường niên khi đạt doanh số giao dịch chi tiết tại đây http www sacombank com vn ca nhan the dich vu the mien giam phi thuong nien html 3 phí chuyển đổi ngoại tệ 1 99 số tiền giao dịch 4 phí khác tham khảo biểu phí tại đây http www sacombank com vn cong cu bieu phi html khcn hạn mức tín dụng từ 40 triệu vnd trở lên tối đa không giới hạn","content_sm_ltks":"thẻ sacombank platinum american express biểu phí 1 phí phát hành miễn phí 2 phí thường niên thẻ chính 499 000 vnd thẻ phụ miễn phí từ năm thứ 2 trở đi chủ thẻ tín dụng được miễn phí thường niên khi đạt doanh số giao dịch chi tiết tại đây http www sacombank com vn ca nhan the dich vu the mien giam phi thuong nien html 3 phí chuyển đổi ngoại tệ 1 99 số tiền giao dịch 4 phí khác tham khảo biểu phí tại đây http www sacombank com vn cong cu bieu phi html khcn hạn mức tín dụng từ 40 triệu vnd trở lên tối đa không giới hạn","img_id":"","doc_id":"3892a0ae95fe11f09f6aff00348f2de6","title_tks":"chapter biểu phí lãi suất","important_kwd":["Thẻ Sacombank Platinum American Express"," Phí thường niên 499.000 VND"," Miễn phí thường niên khi đạt doanh số"," Phí chuyển đổi ngoại tệ 1","99%"," Hạn mức tín dụng từ 40 triệu VND"],"kb_id":"141c05f895fe11f0a3860de5ef2cedf3","position_int":[[2,1,1,1,1]],"important_tks":"thẻ sacombank platinum american express phí thường niên 499 000 vnd miễn phí thường niên khi đạt doanh số phí chuyển đổi ngoại tệ 1 99 hạn mức tín dụng từ 40 triệu vnd","top_int":[1],"docnm_kwd":"Chapter: Biểu phí & Lãi suất.bookstack","page_num_int":[2],"title_sm_tks":"chapter biểu phí lãi suất","question_kwd":["Phí thường niên của thẻ Sacombank Platinum American Express là bao nhiêu và khi nào chủ thẻ được miễn phí?  ","Phí chuyển đổi ngoại tệ áp dụng cho giao dịch bằng thẻ này là bao nhiêu phần trăm?  ","Hạn mức tín dụng tối thiểu và tối đa của thẻ Sacombank Platinum American Express là bao nhiêu?"]}}]}}



  {
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
  'CARD_PRODUCT': ['THẺ TÍN DỤNG SACOMBANK', 'THẺ VISA PLATINUM'],
  'PAYMENT_NETWORK': ['VISA', 'MASTERCARD'],
  'BANK': ['SACOMBANK', 'VIETCOMBANK'],
  'CARD_TIER': ['PLATINUM', 'GOLD'],
  'CASHBACK_FEATURE': ['HOÀN TIỀN 2%', 'CASHBACK KHÔNG GIỚI HẠN'],
  'REWARD_PROGRAM': ['TÍCH ĐIỂM MILES', 'QUÀ TẶNG SINH NHẬT'],
  'INSURANCE_BENEFIT': ['BẢO HIỂM DU LỊCH TOÀN CẦU', 'BẢO HIỂM Y TẾ'],
  'AIRPORT_SERVICE': ['PHÒNG CHỜ CIP', 'FAST TRACK SÂN BAY'],
  'DINING_BENEFIT': ['GIẢM 20% NHÀ HÀNG', 'ƯU ĐÃI F&B'],
  'ANNUAL_FEE': ['PHÍ THƯỜNG NIÊN 1.999.000 VND', 'MIỄN PHÍ NĂM ĐẦU'],
  'INTEREST_RATE': ['LÃI SUẤT 0% 45 NGÀY', 'LÃI SUẤT 2.95%/THÁNG'],
  'CREDIT_LIMIT': ['HẠN MỨC 500 TRIỆU', 'HẠN MỨC TÍN DỤNG CAO'],
  'CURRENCY': ['VND', 'USD'],
  'CUSTOMER_SEGMENT': ['KHÁCH HÀNG VIP', 'SINH VIÊN'],
  'SPENDING_CATEGORY': ['SHOPPING ONLINE', 'DU LỊCH QUỐC TẾ'],
  'MERCHANT_PARTNER': ['TIKI', 'GRAB'],
  'PROMOTION': ['KHUYẾN MÃI TẾT', 'ƯU ĐÃI BLACK FRIDAY'],
  'ELIGIBILITY': ['THU NHẬP TỐI THIỂU 15 TRIỆU', 'TUỔI TỪ 18-65']
}