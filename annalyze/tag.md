để tìm tag => dựa vào câu query (có thẻ đã token) => ES query tập tags kb (full text : text, keyworld, quesion + vector), nó ko tìm trên tag của document đó, sau khi có , sẽ arrigate lại tag của document đó (cộng các tag tìm dc lên count)

=> dựa vào count tính điểm tag (cosine similarity) => chon ra top 3 tag có điểm cao
dựa vào 3 tag này mới đi rank_feas cho tag của từng document.

=> boot và nhân lên điểm số rất cao nếu match
=> điểm giảm nếu có tag ko match (điểm càng cào thì càng giảm), chưa biết tại sao

Hãy xem chỗ tính điểm cho tag trong _rank_feature_scores() function:

⏺ Read(ragflow/rag/nlp/search.py)
  ⎿  Read 50 lines (ctrl+o to expand)

⏺ Read(ragflow/rag/nlp/search.py)
  ⎿  Read 15 lines (ctrl+o to expand)

⏺ Perfect! Đây là công thức tính điểm tag:

  Tag Scoring Formula:

  1. Cosine Similarity Calculation (lines 280-292):

  # For each document
  nor = 0          # Numerator (dot product)
  denor = 0        # Document denominator

  for tag, score in document_tags.items():
      if tag in query_tags:
          nor += query_tags[tag] * score      # Matched tags contribute
      denor += score * score                  # All doc tags contribute to norm

  # Query denominator
  q_denor = sqrt(sum(query_score² for query_score in query_tags))

  # Final tag similarity
  tag_similarity = nor / (sqrt(denor) * q_denor)

  2. Final Rank Score (line 296):

  final_rank_scores = tag_similarities * 10. + pageranks
                      # ↑ TAG BOOST 10X!

  Concrete Example:

  # Query tags: {'Bảo hiểm': 1, 'Cách đăng ký': 1}
  # Doc tags: {'Bảo hiểm': 10, 'Phí thẻ': 5}

  nor = 1 * 10 = 10                    # Only 'Bảo hiểm' matches
  denor = 10² + 5² = 125              # All doc tags
  q_denor = sqrt(1² + 1²) = 1.41      # All query tags

  tag_similarity = 10 / (sqrt(125) * 1.41) = 10 / 15.76 = 0.6348
  final_boost = 0.6348 * 10 = 6.348   # ← HUGE BOOST!

  Key Points:

  1. Line 296: * 10. là chỗ boost tag similarity lên 10 lần!
  2. Cosine similarity: Penalize documents có nhiều irrelevant tags
  3. PageRank: Được cộng thêm (thường nhỏ)

  Để giảm tag boost, sửa line 296:
  final_rank_scores = tag_similarities * 3. + pageranks  # Giảm từ 10x xuống 3x


 def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim
  def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        logging.info(f"[TAG_QUERY] Starting tag extraction for question: '{question[:100]}{'...' if len(question) > 100 else ''}'")
        logging.info(f"[TAG_QUERY] all_tags sample: {dict(list(all_tags.items())[:5]) if all_tags else 'Empty/None'}")

        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]

        match_txt, keywords = self.qryr.question(question, min_match=0.0)

        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")

        if not aggs:
            logging.info(f"[TAG_QUERY] No tag aggregations found")
            return {}

        logging.info(f"[TAG_QUERY] Raw tag aggregations: {aggs[:10]}")

        cnt = np.sum([c for _, c in aggs])
        logging.info(f"[TAG_QUERY] Total tag count: {cnt}")
        logging.info(f"[TAG_QUERY] all_tags type: {type(all_tags)}, size: {len(all_tags) if all_tags else 0}")
        logging.info(f"[TAG_QUERY] all_tags sample: {dict(list(all_tags.items())[:5]) if all_tags else 'Empty'}")

        # Calculate tag scores with better scaling
        tag_scores = []
        for a, c in aggs:
            idf_weight = all_tags.get(a, 0.0001) if all_tags else 0.0001
            tf_component = (c + 1) / (cnt + S)

            # Debug the division step by step
            tf_div_idf = tf_component / max(1e-6, idf_weight)
            raw_score = 0.1 * tf_div_idf

            # Use higher precision scaling to differentiate scores
            scaled_score = raw_score * 10000  # Scale up more
            final_score = max(1, int(scaled_score))  # Use int instead of round for more precision

            tag_scores.append((a, final_score))
            logging.info(f"[TAG_QUERY] Tag '{a}': c={c}, tf={tf_component:.6f}, idf={idf_weight:.6f}, tf/idf={tf_div_idf:.6f}, raw={raw_score:.6f}, final={final_score}")

        tag_fea = sorted(tag_scores, key=lambda x: x[1] * -1)[:topn_tags]

        logging.info(f"[TAG_QUERY] Calculated tag features: {tag_fea}")

        result = {a.replace(".", "_"): max(1, c) for a, c in tag_fea}
        logging.info(f"[TAG_QUERY] Final tag features: {result}")

        return result
