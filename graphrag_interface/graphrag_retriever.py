import os
import json
from collections import defaultdict
from itertools import islice
from typing import Dict, List, Tuple, Optional, Any, Set, Union

import torch
import numpy as np
import networkx as nx
import chromadb


class GraphAugmentedRetriever:
    """
    Image or Text -> ANN -> seed top-M -> graph expansion (relations + co-occurrence) -> rerank -> top-N.

    Returns TWO rankings per query (no evaluation inside):
      - 'baseline': CLIP-only list with embedding similarities
      - 'graph'   : graph-augmented list with blended score and components
    """
    # Lightweight tuning knobs
    FAST_POSTINGS_ONLY = True   # use postings for scoring; avoid per-candidate GraphML reads
    FILTER_ANCHOR_BY_DF = True  # drop ultra-rare / ultra-common anchor tokens
    DF_MIN = 5                  # keep tokens seen in at least 5 studies
    DF_MAX = 3000               # ...but not too common
    DEG_MAX_REL = 2000          # skip anchors with huge degree (relation graph)
    DEG_MAX_COOC = 2000         # skip anchors with huge degree (co-occ graph)

    def __init__(
        self,
        model_wrapper,
        chroma_dir: str,
        image_collection_name: str,
        text_collection_name: str,
        local_graph_dir: str,
        global_graph_dir: str,
        postings_dir: str,
        global_rel_filename: str = "global_relations.graphml",
        global_cooc_filename: str = "global_cooc.graphml",
    ):
        # -------- Model --------
        self.model = model_wrapper.model.eval().cuda()
        self.tokenizer = model_wrapper.tokenizer
        self.context_length = model_wrapper.train_cfg["context_length"]

        # Optional hint for dummy image (used for text-only encode fallback)
        self._input_res = None
        for k in ("image_size", "input_size", "resolution"):
            if k in model_wrapper.train_cfg:
                v = model_wrapper.train_cfg[k]
                # accept 224 or (224, 224) etc.
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    self._input_res = int(v[0])
                elif isinstance(v, int):
                    self._input_res = v
                break

        # -------- Vector DB --------
        self.chroma = chromadb.PersistentClient(path=chroma_dir)
        self.image_col = self.chroma.get_or_create_collection(image_collection_name)
        self.text_col = self.chroma.get_or_create_collection(text_collection_name)

        # -------- Paths --------
        self.local_graph_dir = local_graph_dir
        self.global_rel_path = os.path.join(global_graph_dir, global_rel_filename)
        self.global_cooc_path = os.path.join(global_graph_dir, global_cooc_filename)

        self.postings_dir = postings_dir
        self.token_postings_path = os.path.join(postings_dir, "token_postings.jsonl")
        self.relation_postings_path = os.path.join(postings_dir, "relation_postings.jsonl")
        self.cooc_postings_path = os.path.join(postings_dir, "cooc_postings.jsonl")

        # -------- Load global graphs once --------
        self.G_rel = nx.read_graphml(self.global_rel_path)
        if not isinstance(self.G_rel, nx.MultiDiGraph):
            self.G_rel = nx.MultiDiGraph(self.G_rel)

        self.G_cooc = nx.read_graphml(self.global_cooc_path)
        if not isinstance(self.G_cooc, nx.Graph) or isinstance(self.G_cooc, nx.DiGraph):
            self.G_cooc = nx.Graph(self.G_cooc)

        # -------- Lazy postings --------
        self._token_postings = None        # token -> [study_ids]
        self._relation_postings = None     # (src, rel, dst) -> [study_ids]
        self._cooc_postings = None         # (min_tok, max_tok) -> [study_ids] (optional)

        # --- Node id <-> token maps
        self.rel_id_to_token = {str(n): str(d.get("token", "")).lower() for n, d in self.G_rel.nodes(data=True)}
        self.rel_token_to_id = {tok: nid for nid, tok in self.rel_id_to_token.items() if tok}

        self.cooc_id_to_token = {str(n): str(d.get("token", "")).lower() for n, d in self.G_cooc.nodes(data=True)}
        self.cooc_token_to_id = {tok: nid for nid, tok in self.cooc_id_to_token.items() if tok}

        # Small in-memory cache to avoid re-reading same study files in one run
        self._local_cache_tokens: Dict[str, set] = {}      # sid -> set[str]
        self._local_cache_relations: Dict[str, set] = {}   # sid -> set[(src, rel, dst)]

        # Dummy image cache for text-only path (allocated on demand)
        self._dummy_img: Optional[torch.Tensor] = None

    # =========================
    # Public API
    # =========================
    @torch.no_grad()
    def retrieve_pairs(
        self,
        query_image_tensor: Optional[torch.Tensor] = None,
        *,
        query_text: Optional[Union[str, torch.Tensor]] = None,
        K: int = 200,
        M: int = 30,
        limits: Optional[Dict[str, int]] = None,
        weights: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        topN: int = 50,
        normalize_scores: bool = True,
        include_text_docs: bool = False,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Image→RAG or Text→RAG retrieval with optional graph augmentation.

        - If `query_image_tensor` is provided: image ANN on image collection.
        - Else if `query_text` is provided (str or token tensor): text ANN on text collection.
        - Otherwise: returns empty result.

        Returns
        -------
        dict
            {
              "baseline": [ ... ],
              "graph":    [ ... ],
              (optional) "debug": {...}
            }
        """
        # 0) Defaults
        limits = limits or {"k_rel_per_anchor_dir": 100, "k_cooc_per_anchor": 200, "pool_cap": 3000}
        alpha, beta, gamma = weights

        # 1) ANN (CLIP → top-K)
        if query_image_tensor is not None:
            q = self._encode_query_image(query_image_tensor)
            C0 = self._query_chroma_images(q, K)  # [(sid, sim)]
        elif query_text is not None:
            q_text_emb = self._encode_query_text(query_text)
            if q_text_emb is None:
                ret = {"baseline": [], "graph": []}
                if return_debug:
                    ret["debug"] = {"C0": [], "anchors": {"tokens": [], "relations": []}, "reason": "no_text_embedding"}
                return ret
            C0 = self._query_chroma_texts(q_text_emb, K)
        else:
            ret = {"baseline": [], "graph": []}
            if return_debug:
                ret["debug"] = {"C0": [], "anchors": {"tokens": [], "relations": []}, "reason": "no_query"}
            return ret

        if not C0:
            ret = {"baseline": [], "graph": []}
            if return_debug:
                ret["debug"] = {"C0": [], "anchors": {"tokens": [], "relations": []}}
            return ret

        # 2) Warm local token/relation cache for K
        cand_tokens, cand_relations = self._warm_local_cache(C0)

        # 3) Build anchor context from top-M
        Seeds = C0[: min(M, len(C0))]
        AnchorTokens, AnchorRelations = self._build_anchor_context(Seeds, cand_tokens, cand_relations)
        AnchorTokens = self._thin_anchors(AnchorTokens)

        # 4) Graph expansion (global graphs + postings)
        expanded_ids = self._expand_candidates(anchor_tokens=AnchorTokens, limits=limits)

        # 5) Build rerank pool (cap size)
        pool_ids = self._build_pool(C0, expanded_ids, pool_cap=limits["pool_cap"])

        # Ensure all pool ids have local tokens/relations
        self._ensure_local_for_pool(pool_ids, cand_tokens, cand_relations)

        # 6) Score & rerank pool
        sim_map = {sid: sim for sid, sim in C0}
        graph_scores = self._score_candidates(
            candidate_ids=list(pool_ids),
            sim_map=sim_map,
            cand_tokens=cand_tokens,
            cand_relations=cand_relations,
            anchor_tokens=AnchorTokens,
            normalize=normalize_scores,
            weights=(alpha, beta, gamma),
        )
        ranked = sorted(graph_scores.items(), key=lambda kv: kv[1]["final"], reverse=True)

        # 7) Batch-fetch metadata/docs for pool ids (one pass)
        img_meta_map, text_meta_map, text_doc_map = self._batch_fetch_meta(pool_ids, include_text_docs)

        # 8) Package outputs
        baseline_ranked = self._package_baseline(
            C0=C0, topN=topN, img_meta_map=img_meta_map, text_meta_map=text_meta_map, text_doc_map=text_doc_map,
        )
        graph_ranked = self._package_graph(
            ranked=ranked, topN=topN, img_meta_map=img_meta_map, text_meta_map=text_meta_map, text_doc_map=text_doc_map,
        )

        # === NEW: hydrate refined text for the Top-N only ===
        top_ids = list({*(i["study_id"] for i in baseline_ranked), *(i["study_id"] for i in graph_ranked)})
        doc_map = self._hydrate_docs_for_topN(top_ids)

        def _attach_docs(items):
            for it in items:
                sid = it["study_id"]
                txt = doc_map.get(sid, "")
                if not txt:
                    # fall back to reading from the path in metadata (robust even if Chroma docs missing)
                    txt = self._read_text_file(it.get("refined_report_txt_path", ""))
                it["refined_report_txt"] = txt  # <-- attach the string

        _attach_docs(baseline_ranked)
        _attach_docs(graph_ranked)
        # ===============================================

        out: Dict[str, Any] = {"baseline": baseline_ranked, "graph": graph_ranked}
        if return_debug:
            out["debug"] = {
                "C0": [{"study_id": sid, "sim_emb": sim} for sid, sim in C0],
                "anchors": {"tokens": sorted(list(AnchorTokens)), "relations": sorted(list(AnchorRelations))},
                "expanded_ids": sorted(list(expanded_ids)),
                "mode": "image" if query_image_tensor is not None else "text",
            }
        return out

    # =========================
    # Stage 1: ANN retrieval
    # =========================
    def _encode_query_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Encode query image to CLIP image features (L2-normalized).
        Required model signature: image_features, text_features, logit_scale = self.model(images, tokens)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.cuda()

        dummy_tokens = torch.zeros((image_tensor.shape[0], self.context_length),
                                   dtype=torch.long, device=image_tensor.device)

        image_features, _, _ = self.model(image_tensor, dummy_tokens)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
        return image_features.detach().cpu().numpy().astype(np.float32)

    def _encode_query_text(self, query_text: Union[str, torch.Tensor]) -> Optional[np.ndarray]:
        """
        Encode query text to CLIP text features (L2-normalized).
        Accepts raw string or a pre-tokenized LongTensor of shape [1, context_length].
        """
        try:
            if isinstance(query_text, torch.Tensor):
                tokens = query_text
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                tokens = tokens.to(dtype=torch.long, device="cuda")
            else:
                # raw string: tokenize
                tokens = self.tokenizer(str(query_text), self.context_length)
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                tokens = tokens.to(device="cuda", dtype=torch.long)

            # Fast path: model has encode_text
            if hasattr(self.model, "encode_text"):
                text_features = self.model.encode_text(tokens)
            else:
                # Fallback: call forward(images, tokens) with a dummy image
                dummy_img = self._get_dummy_image(batch=tokens.shape[0]).to(tokens.device)
                _, text_features, _ = self.model(dummy_img, tokens)

            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)
            return text_features.detach().cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def _get_dummy_image(self, batch: int = 1) -> torch.Tensor:
        """Create a dummy image tensor for text-only forward fallback."""
        H = W = self._input_res or 224
        return torch.zeros((batch, 3, H, W), dtype=torch.float32)

    def _query_chroma_images(self, q_img: np.ndarray, K: int) -> List[Tuple[str, float]]:
        """
        Query Chroma image collection. Return [(study_id, sim_emb)] sorted desc by similarity.
        Converts distances to similarities if collection uses cosine distance.
        """
        res = self.image_col.query(query_embeddings=q_img.tolist(), n_results=K, include=["distances"])
        ids = self._first_or_self(res.get("ids", [])) or []
        distances = self._first_or_self(res.get("distances", [])) or []
        sims = [1.0 - float(d) for d in distances]  # cosine distance -> similarity
        pairs = list(zip([str(i) for i in ids], sims))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    def _query_chroma_texts(self, q_txt: np.ndarray, K: int) -> List[Tuple[str, float]]:
        """
        Query Chroma text collection. Return [(study_id, sim_emb)] sorted desc by similarity.
        """
        res = self.text_col.query(query_embeddings=q_txt.tolist(), n_results=K, include=["distances"])
        ids = self._first_or_self(res.get("ids", [])) or []
        distances = self._first_or_self(res.get("distances", [])) or []
        sims = [1.0 - float(d) for d in distances]
        pairs = list(zip([str(i) for i in ids], sims))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    # =========================
    # Stage 2: Local graphs
    # =========================
    def _warm_local_cache(self, C0: List[Tuple[str, float]]) -> Tuple[Dict[str, set], Dict[str, set]]:
        cand_tokens: Dict[str, set] = {}
        cand_relations: Dict[str, set] = {}
        for sid, _ in C0:
            g_path = os.path.join(self.local_graph_dir, f"{sid}.graphml")
            tokens, rels = self._extract_tokens_relations_from_local_graph(g_path)
            cand_tokens[sid] = tokens
            cand_relations[sid] = rels
        return cand_tokens, cand_relations

    def _extract_tokens_relations_from_local_graph(self, graph_path: str):
        """
        Load per-study tokens/relations from GraphML, with a small in-memory cache.
        """
        sid = os.path.splitext(os.path.basename(graph_path))[0]

        # cache hit
        if sid in self._local_cache_tokens:
            return self._local_cache_tokens[sid], self._local_cache_relations[sid]

        # missing file
        if not os.path.exists(graph_path):
            self._local_cache_tokens[sid] = set()
            self._local_cache_relations[sid] = set()
            return set(), set()

        # read GraphML
        try:
            G = nx.read_graphml(graph_path)
        except Exception:
            self._local_cache_tokens[sid] = set()
            self._local_cache_relations[sid] = set()
            return set(), set()

        tokens, rels = set(), set()

        # nodes -> tokens
        for _, attrs in G.nodes(data=True):
            t = str(attrs.get("token_lower", "")).lower()
            if t:
                tokens.add(t)

        # edges -> typed triples
        for u, v, attrs in G.edges(data=True):
            rel = str(attrs.get("type", ""))
            su = str(G.nodes[u].get("token_lower", "")).lower()
            sv = str(G.nodes[v].get("token_lower", "")).lower()
            if su and sv and rel:
                rels.add((su, rel, sv))

        # cache & return
        self._local_cache_tokens[sid] = tokens
        self._local_cache_relations[sid] = rels
        return tokens, rels

    def _build_anchor_context(
        self,
        Seeds: List[Tuple[str, float]],
        cand_tokens: Dict[str, set],
        cand_relations: Dict[str, set],
    ) -> Tuple[set, set]:
        anchor_tokens, anchor_relations = set(), set()
        for sid, _ in Seeds:
            anchor_tokens.update(cand_tokens.get(sid, set()))
            anchor_relations.update(cand_relations.get(sid, set()))
        return anchor_tokens, anchor_relations

    def _thin_anchors(self, AnchorTokens: set) -> set:
        """Apply DF and degree caps to anchors (optional)."""
        if not (self.FILTER_ANCHOR_BY_DF or self.DEG_MAX_REL or self.DEG_MAX_COOC):
            return AnchorTokens
        self._ensure_postings_loaded()

        filtered = set()
        for t in AnchorTokens:
            # df filter
            if self.FILTER_ANCHOR_BY_DF:
                df = len(self._token_postings.get(t, []))
                if df < self.DF_MIN or df > self.DF_MAX:
                    continue
            # degree caps
            nid_rel = self.rel_token_to_id.get(t)
            if self.DEG_MAX_REL and nid_rel is not None:
                if (self.G_rel.out_degree(nid_rel) + self.G_rel.in_degree(nid_rel)) > self.DEG_MAX_REL:
                    continue
            nid_cooc = self.cooc_token_to_id.get(t)
            if self.DEG_MAX_COOC and nid_cooc is not None:
                if self.G_cooc.degree(nid_cooc) > self.DEG_MAX_COOC:
                    continue
            filtered.add(t)
        return filtered

    # =========================
    # Stage 4: Expansion
    # =========================
    def _expand_candidates(self, anchor_tokens: set, limits: Dict[str, int]) -> set:
        """
        Expand candidate study_ids using global graph + postings.
        """
        self._ensure_postings_loaded()
        rel_neighbors = self._top_rel_neighbors(anchor_tokens, limits["k_rel_per_anchor_dir"])
        cooc_neighbors = self._top_cooc_neighbors(anchor_tokens, limits["k_cooc_per_anchor"])

        expanded_ids = set()
        # Relation postings
        for (src, rel, dst) in rel_neighbors:
            expanded_ids.update(self._relation_postings.get((src, rel, dst), []))
        # Co-occurrence postings (via token postings)
        for tok in cooc_neighbors:
            expanded_ids.update(self._token_postings.get(tok, []))
        return expanded_ids

    def _top_rel_neighbors(self, anchor_tokens: set, k_per_dir: int) -> set:
        """
        Collect top-k per-direction relation triples (src, rel, dst) for each anchor token.
        """
        triples = set()
        for t in anchor_tokens:
            nid = self.rel_token_to_id.get(t)
            if nid is None:
                continue
            if self.DEG_MAX_REL and (self.G_rel.out_degree(nid) + self.G_rel.in_degree(nid)) > self.DEG_MAX_REL:
                continue
            # Outgoing
            out_edges = []
            for _, v, key, data in self.G_rel.out_edges(nid, keys=True, data=True):
                w = int(data.get("weight", 1))
                rel = str(data.get("type", key))
                out_edges.append((nid, rel, v, w))
            out_edges.sort(key=lambda x: x[3], reverse=True)
            for u, rel, v, _ in out_edges[:k_per_dir]:
                triples.add((self.rel_id_to_token[u], rel, self.rel_id_to_token[str(v)]))
            # Incoming
            in_edges = []
            for u, _, key, data in self.G_rel.in_edges(nid, keys=True, data=True):
                w = int(data.get("weight", 1))
                rel = str(data.get("type", key))
                in_edges.append((u, rel, nid, w))
            in_edges.sort(key=lambda x: x[3], reverse=True)
            for u, rel, v, _ in in_edges[:k_per_dir]:
                triples.add((self.rel_id_to_token[str(u)], rel, self.rel_id_to_token[v]))
        return triples

    def _top_cooc_neighbors(self, anchor_tokens: set, k_per_anchor: int) -> set:
        """
        Collect top-k co-occurrence neighbors per anchor token by weight.
        """
        neighbors = set()
        for t in anchor_tokens:
            nid = self.cooc_token_to_id.get(t)
            if nid is None:
                continue
            if self.DEG_MAX_COOC and self.G_cooc.degree(nid) > self.DEG_MAX_COOC:
                continue
            neigh = []
            for nb in self.G_cooc.neighbors(nid):
                data = self.G_cooc.get_edge_data(nid, nb) or {}
                w = float(data.get("weight", 1.0))
                neigh.append((nb, w))
            neigh.sort(key=lambda x: x[1], reverse=True)
            for nb, _ in neigh[:k_per_anchor]:
                neighbors.add(self.cooc_id_to_token[str(nb)])
        return neighbors

    def _ensure_postings_loaded(self):
        """Lazy-load postings JSONL into memory maps."""
        if self._token_postings is None:
            self._token_postings = defaultdict(list)
            with open(self.token_postings_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._token_postings[rec["token"]] = rec["studies"]

        if self._relation_postings is None:
            self._relation_postings = defaultdict(list)
            with open(self.relation_postings_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    key = (rec["src"], rec["rel"], rec["dst"])
                    self._relation_postings[key] = rec["studies"]

        if self._cooc_postings is None:
            self._cooc_postings = defaultdict(list)
            if os.path.exists(self.cooc_postings_path):
                with open(self.cooc_postings_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        u, v = rec["u"], rec["v"]
                        key = (min(u, v), max(u, v))
                        self._cooc_postings[key] = rec["studies"]

    # =========================
    # Stage 5: Scoring + Reranking
    # =========================
    def _score_candidates(
        self,
        candidate_ids: List[str],
        sim_map: Dict[str, float],
        cand_tokens: Dict[str, set],
        cand_relations: Dict[str, set],
        anchor_tokens: set,
        normalize: bool,
        weights: Tuple[float, float, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute emb/rel/cooc components and blended final score.
        If FAST_POSTINGS_ONLY=True, we score rel/cooc using postings membership only.
        """
        alpha, beta, gamma = weights

        # Embedding component (0 if not in C0)
        emb_scores = {sid: float(sim_map.get(sid, 0.0)) for sid in candidate_ids}

        # Build relation weights around anchors
        selected_rel = self._top_rel_neighbors(anchor_tokens, k_per_dir=1000)
        rel_weight_map: Dict[Tuple[str, str, str], int] = {}
        for (u_tok, rel, v_tok) in selected_rel:
            u_id = self.rel_token_to_id.get(u_tok)
            v_id = self.rel_token_to_id.get(v_tok)
            if u_id is None or v_id is None:
                continue
            data = self.G_rel.get_edge_data(u_id, v_id)
            w_sum = 0
            if data:
                for k, d in data.items():
                    if str(d.get("type", k)) == rel:
                        w_sum += int(d.get("weight", 1))
            if w_sum > 0:
                rel_weight_map[(u_tok, rel, v_tok)] = w_sum

        # Co-occ weights to anchors
        from collections import defaultdict as _dd
        cooc_to_anchor = _dd(float)
        for a_tok in anchor_tokens:
            a_id = self.cooc_token_to_id.get(a_tok)
            if a_id is None:
                continue
            for nb in self.G_cooc.neighbors(a_id):
                d = self.G_cooc.get_edge_data(a_id, nb) or {}
                w = float(d.get("weight", 1.0))
                nb_tok = self.cooc_id_to_token[str(nb)]
                cooc_to_anchor[nb_tok] += np.log1p(w)

        if self.FAST_POSTINGS_ONLY:
            self._ensure_postings_loaded()
            # RELATION via postings
            rel_scores = {sid: 0.0 for sid in candidate_ids}
            for triple, w_sum in rel_weight_map.items():
                inc = float(np.log1p(w_sum))
                for sid in self._relation_postings.get(triple, []):
                    if sid in rel_scores:
                        rel_scores[sid] += inc
            # COOC via postings
            cooc_scores = {sid: 0.0 for sid in candidate_ids}
            for tok, inc in cooc_to_anchor.items():
                for sid in self._token_postings.get(tok, []):
                    if sid in cooc_scores:
                        cooc_scores[sid] += float(inc)
        else:
            # Precise path: per-candidate tokens/relations
            rel_scores = {}
            for sid in candidate_ids:
                s = 0.0
                for triple in cand_relations.get(sid, set()):
                    if triple in rel_weight_map:
                        s += np.log1p(rel_weight_map[triple])
                rel_scores[sid] = float(s)

            cooc_scores = {}
            for sid in candidate_ids:
                s = 0.0
                for tok in cand_tokens.get(sid, set()):
                    s += cooc_to_anchor.get(tok, 0.0)
                cooc_scores[sid] = float(s)

        # Normalize
        if normalize:
            emb_norm = self._minmax(emb_scores)
            rel_norm = self._minmax(rel_scores)
            cooc_norm = self._minmax(cooc_scores)
        else:
            emb_norm, rel_norm, cooc_norm = emb_scores, rel_scores, cooc_scores

        out = {}
        for sid in candidate_ids:
            final = alpha * emb_norm.get(sid, 0.0) + beta * rel_norm.get(sid, 0.0) + gamma * cooc_norm.get(sid, 0.0)
            out[sid] = {
                "final": float(final),
                "score_emb_norm": float(emb_norm.get(sid, 0.0)),
                "score_rel_norm": float(rel_norm.get(sid, 0.0)),
                "score_cooc_norm": float(cooc_norm.get(sid, 0.0)),
                # keep backward-compat keys used by your print code
                "emb_norm": float(emb_norm.get(sid, 0.0)),
                "rel_norm": float(rel_norm.get(sid, 0.0)),
                "cooc_norm": float(cooc_norm.get(sid, 0.0)),
            }
        return out

    # =========================
    # Packaging / helpers
    # =========================
    def _build_pool(self, C0: List[Tuple[str, float]], expanded_ids: Set[str], pool_cap: int) -> Set[str]:
        pool_ids = set([sid for sid, _ in C0])
        pool_ids.update(expanded_ids)
        if len(pool_ids) > pool_cap:
            expand_only = [pid for pid in pool_ids if pid not in [sid for sid, _ in C0]]
            keep = set([sid for sid, _ in C0] + list(islice(expand_only, pool_cap - len(C0))))
            pool_ids = keep
        return pool_ids

    def _ensure_local_for_pool(self, pool_ids: Set[str], cand_tokens: Dict[str, set], cand_relations: Dict[str, set]) -> None:
        for sid in list(pool_ids):
            if sid not in cand_tokens:
                g_path = os.path.join(self.local_graph_dir, f"{sid}.graphml")
                tokens, rels = self._extract_tokens_relations_from_local_graph(g_path)
                cand_tokens[sid] = tokens
                cand_relations[sid] = rels

    def _batch_fetch_meta(self, pool_ids: Set[str], include_text_docs: bool) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]
    ]:
        """Return (img_meta_map, text_meta_map, text_doc_map)."""
        pool_ids_list = list(pool_ids)

        img_meta_map: Dict[str, Dict[str, Any]] = {}
        if pool_ids_list:
            res_img = self.image_col.get(ids=pool_ids_list, include=["metadatas"])
            got_ids = self._first_or_self(res_img.get("ids", [])) or []
            metas = self._first_or_self(res_img.get("metadatas", [])) or []
            for sid, md in zip(got_ids, metas):
                img_meta_map[str(sid)] = md or {}

        text_doc_map: Dict[str, str] = {}
        text_meta_map: Dict[str, Dict[str, Any]] = {}
        if pool_ids_list:
            include = ["metadatas", "documents"] if include_text_docs else ["metadatas"]
            res_txt = self.text_col.get(ids=pool_ids_list, include=include)
            got_ids_t = self._first_or_self(res_txt.get("ids", [])) or []
            metas_t = self._first_or_self(res_txt.get("metadatas", [])) or []
            docs_t = self._first_or_self(res_txt.get("documents", [])) or []
            for idx, sid in enumerate(got_ids_t):
                sid = str(sid)
                text_meta_map[sid] = metas_t[idx] if idx < len(metas_t) and metas_t[idx] else {}
                if include_text_docs and idx < len(docs_t):
                    text_doc_map[sid] = docs_t[idx] or ""

        return img_meta_map, text_meta_map, text_doc_map

    def _package_baseline(
            self,
            C0,
            topN,
            img_meta_map,
            text_meta_map,
            text_doc_map,  # kept in signature for compatibility; not used
    ):
        out = []
        for rank, (sid, sim) in enumerate(C0[:topN], start=1):
            sid_str = str(sid)
            image_path = (img_meta_map.get(sid_str, {}) or {}).get("image_path", "")
            refined_path = self._resolve_refined_path(text_meta_map.get(sid_str, {}))
            out.append({
                "study_id": sid_str,
                "rank": rank,
                "score_emb": sim,
                "local_graph_path": os.path.join(self.local_graph_dir, f"{sid_str}.graphml"),
                "image_path": image_path,
                "refined_report_txt_path": refined_path,
            })
        return out

    def _package_graph(
            self,
            ranked,
            topN,
            img_meta_map,
            text_meta_map,
            text_doc_map,  # kept in signature for compatibility; not used
    ):
        out = []
        for rank, (sid, comp) in enumerate(ranked[:topN], start=1):
            sid_str = str(sid)
            image_path = (img_meta_map.get(sid_str, {}) or {}).get("image_path", "")
            refined_path = self._resolve_refined_path(text_meta_map.get(sid_str, {}))
            out.append({
                "study_id": sid_str,
                "rank": rank,
                "final_score": comp["final"],
                "score_emb_norm": comp["emb_norm"],
                "score_rel_norm": comp["rel_norm"],
                "score_cooc_norm": comp["cooc_norm"],
                "local_graph_path": os.path.join(self.local_graph_dir, f"{sid_str}.graphml"),
                "image_path": image_path,
                "refined_report_txt_path": refined_path,
            })
        return out

    # =========================
    # Utils
    # =========================
    @staticmethod
    def _first_or_self(x):
        """Chroma sometimes returns nested lists: [[...]]; normalize that."""
        return x[0] if isinstance(x, list) and x and isinstance(x[0], list) else x

    @staticmethod
    def _minmax(d: Dict[str, float]) -> Dict[str, float]:
        arr = np.array(list(d.values()), dtype=np.float64)
        if arr.size == 0:
            return {k: 0.0 for k in d}
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax <= vmin + 1e-12:
            return {k: 0.0 for k in d}
        denom = (vmax - vmin)
        return {k: (v - vmin) / denom for k, v in d.items()}

    # =========================
    # Helper: fetch text doc (kept for backward compat, but batch path is preferred)
    # =========================
    def _get_text_doc(self, study_id: str) -> str:
        try:
            res = self.text_col.get(ids=[study_id], include=["documents"])
            docs = self._first_or_self(res.get("documents", [])) or []
            return docs[0] if docs else ""
        except Exception:
            return ""

    def _hydrate_docs_for_topN(self, ids: list[str]) -> dict[str, str]:
        """
        Fetch Chroma 'documents' for a small set of study_ids.
        Returns {sid: refined_report_txt or ""}.
        """
        if not ids:
            return {}
        res = self.text_col.get(ids=ids, include=["documents"])
        got_ids = self._first_or_self(res.get("ids", [])) or []
        docs = self._first_or_self(res.get("documents", [])) or []
        out = {}
        for k, sid in enumerate(got_ids):
            sid_str = str(sid)
            txt = docs[k] if k < len(docs) and docs[k] else ""
            out[sid_str] = txt
        return out

    def _read_text_file(self, path: str) -> str:
        try:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
        except Exception:
            pass
        return ""

    def _resolve_refined_path(self, meta: dict) -> str:
        meta = meta or {}
        return (
                meta.get("refined_report_txt_path")
                or ""
        )

