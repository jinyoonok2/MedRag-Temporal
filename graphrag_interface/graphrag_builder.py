import os
import json
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm
import torch
import chromadb
import networkx as nx

class GraphRAGBuilder:
    def __init__(
        self,
        model_wrapper,
        dataloader,
        image_collection_name,
        text_collection_name,
        chroma_dir,
        local_graph_dir,
        global_graph_dir,
        postings_dir
    ):
        # Model + tokenizer
        self.model = model_wrapper.model.eval().cuda()
        self.tokenizer = getattr(model_wrapper, "tokenizer", None)

        # Data
        self.dataloader = dataloader

        # ---- ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.image_collection = self.chroma_client.get_or_create_collection(name=image_collection_name)
        self.text_collection = self.chroma_client.get_or_create_collection(name=text_collection_name)

        # ---- Local subgraphs folder
        self.local_graph_dir = local_graph_dir
        os.makedirs(self.local_graph_dir, exist_ok=True)

        # ---- Global graph folder
        self.global_graph_dir = global_graph_dir
        os.makedirs(self.global_graph_dir, exist_ok=True)
        self.global_cooc_path = os.path.join(self.global_graph_dir, "global_cooc.graphml")
        self.global_rel_path = os.path.join(self.global_graph_dir, "global_relations.graphml")

        # ---- Postings folder
        self.postings_dir = postings_dir
        os.makedirs(self.postings_dir, exist_ok=True)
        self.token_postings_path = os.path.join(self.postings_dir, "token_postings.jsonl")
        self.relation_postings_path = os.path.join(self.postings_dir, "relation_postings.jsonl")
        self.cooc_postings_path = os.path.join(self.postings_dir, "cooc_postings.jsonl")

    # ---------------------------
    # Step 1: Save embeddings
    # ---------------------------
    def save_embeddings(self):
        print("Saving image/text embeddings to ChromaDB...")

        for batch in tqdm(self.dataloader):
            images = batch['image_pt'].cuda()
            tokens = batch['refined_report_token'].cuda()
            study_ids = batch['study_id']
            report_txts = batch['refined_report_txt']
            report_paths = batch['refined_report_txt_path']
            image_paths = batch['image_path']

            with torch.no_grad():
                # you required this signature:
                image_features, text_features, logit_scale = self.model(images, tokens)

            for i, study_id in enumerate(study_ids):
                # Image embedding
                self.image_collection.add(
                    documents=[f"image:{study_id}"],
                    embeddings=image_features[i].cpu().tolist(),
                    ids=[study_id],
                    metadatas=[{
                        "study_id": study_id,
                        "image_path": image_paths[i],
                        "modality": "image"
                    }]
                )

                # Text embedding (use actual report text as the document)
                self.text_collection.add(
                    documents=[report_txts[i]],
                    embeddings=text_features[i].cpu().tolist(),
                    ids=[study_id],
                    metadatas=[{
                        "study_id": study_id,
                        "refined_report_txt_path": report_paths[i],
                        "modality": "text"
                    }]
                )

        print("Done. Embeddings persisted to ChromaDB.")

    # ------------------------------------------------------
    # Step 2: Build and save local subgraphs with NetworkX
    # ------------------------------------------------------
    def build_and_save_local_subgraphs(self):
        """
        Build a NetworkX DiGraph per study from its RadGraph JSON and save to GraphML.
        - Stores study_id at graph-level and on vertices/edges.
        - Stores entity metadata and text span info.
        """
        print(f"Building and saving local RadGraph subgraphs (NetworkX) into: {self.local_graph_dir}")

        for batch in tqdm(self.dataloader):
            study_ids = batch["study_id"]
            graph_json_paths = batch["graph_json_path"]

            # Optional extras if the dataloader provides them
            report_text_paths = batch.get("refined_report_txt_path", [None] * len(study_ids))
            image_paths = batch.get("image_path", [None] * len(study_ids))
            report_texts = batch.get("refined_report_txt", [None] * len(study_ids))

            for study_id, graph_json_path, report_text_path, image_path, report_text in zip(
                    study_ids, graph_json_paths, report_text_paths, image_paths, report_texts
            ):
                # Load RadGraph annotation JSON
                with open(graph_json_path, "r", encoding="utf-8") as fh:
                    annotation = json.load(fh)

                entities_dict = annotation.get("entities", {})

                # Create a directed graph
                local_graph = nx.DiGraph()

                # Graph-level attributes
                local_graph.graph["study_id"] = str(study_id)
                local_graph.graph["source"] = "RadGraph"
                local_graph.graph["graph_json_path"] = str(graph_json_path)
                if report_text_path is not None:
                    local_graph.graph["refined_report_txt_path"] = str(report_text_path)
                if image_path is not None:
                    local_graph.graph["image_path"] = str(image_path)
                if report_text is not None:
                    local_graph.graph["refined_report_txt_sample"] = str(report_text[:500])  # keep it short

                # Add vertices (entities)
                for entity_local_id, entity_payload in entities_dict.items():
                    tokens = entity_payload.get("tokens", "")
                    label = entity_payload.get("label", "")
                    start_ix = entity_payload.get("start_ix", None)
                    end_ix = entity_payload.get("end_ix", None)

                    node_attrs = {
                        # Graph internal / linking info
                        "study_id": str(study_id),
                        "entity_id": str(entity_local_id),

                        # Entity metadata
                        "tokens": str(tokens) if tokens is not None else "",
                        "token_lower": str(tokens).lower() if isinstance(tokens, str) else str(tokens),
                        "label": str(label) if label is not None else "",

                        # Text span info
                        "start_ix": -1 if start_ix is None else int(start_ix),
                        "end_ix": -1 if end_ix is None else int(end_ix),
                    }
                    local_graph.add_node(str(entity_local_id), **node_attrs)

                # Add edges (relations)
                for entity_local_id, entity_payload in entities_dict.items():
                    relations_list = entity_payload.get("relations", [])
                    for relation_item in relations_list:
                        # Expected shape: ["relation_type", "target_entity_id"]
                        if not isinstance(relation_item, (list, tuple)) or len(relation_item) != 2:
                            continue
                        relation_type, target_entity_id = relation_item

                        # Only add if both endpoints exist
                        if str(entity_local_id) in local_graph.nodes and str(target_entity_id) in local_graph.nodes:
                            edge_attrs = {
                                "type": str(relation_type),
                                "study_id": str(study_id),
                                "source_entity_id": str(entity_local_id),
                                "target_entity_id": str(target_entity_id),
                            }
                            local_graph.add_edge(str(entity_local_id), str(target_entity_id), **edge_attrs)

                # Persist subgraph (GraphML)
                output_path = os.path.join(self.local_graph_dir, f"{study_id}.graphml")
                nx.write_graphml(local_graph, output_path)

        print("Done. Local subgraphs saved (GraphML).")

    # -----------------------------------------------------------------------
    # NEW: Step 2.5 Build postings to connect global graph back to studies
    # -----------------------------------------------------------------------
    def build_postings_from_subgraphs(self):
        """
        Build inverted indices (postings) that connect global tokens/edges back to study_ids.
        Writes three JSONL files in self.postings_dir.
        """
        print(f"Building postings (token / relation / co-occurrence) from local subgraphs in: {self.local_graph_dir}")

        token_postings = defaultdict(set)  # token -> {study_ids}
        relation_postings = defaultdict(set)  # (src_token, rel_type, dst_token) -> {study_ids}
        cooc_postings = defaultdict(set)  # (min_token, max_token) -> {study_ids}

        local_files = [f for f in os.listdir(self.local_graph_dir) if f.endswith(".graphml")]
        for filename in tqdm(local_files):
            subgraph_path = os.path.join(self.local_graph_dir, filename)
            try:
                local_graph = nx.read_graphml(subgraph_path)
            except Exception as e:
                print(f"Warning: failed to read {subgraph_path}: {e}")
                continue

            study_id = str(local_graph.graph.get("study_id", os.path.splitext(filename)[0]))

            # --- Collect tokens in this study
            tokens_this_study = []
            for _, node_attrs in local_graph.nodes(data=True):
                token_lower = str(node_attrs.get("token_lower", "")).lower()
                if not token_lower:
                    continue
                token_postings[token_lower].add(study_id)
                tokens_this_study.append(token_lower)

            # --- Relations in this study
            for u, v, edge_attrs in local_graph.edges(data=True):
                src_token = str(local_graph.nodes[u].get("token_lower", "")).lower()
                dst_token = str(local_graph.nodes[v].get("token_lower", "")).lower()
                rel_type = str(edge_attrs.get("type", ""))
                if src_token and dst_token and rel_type:
                    relation_postings[(src_token, rel_type, dst_token)].add(study_id)

            # --- Co-occurrence pairs (unique tokens per study to avoid double counting)
            uniq_tokens = sorted(set(tokens_this_study))
            for a, b in combinations(uniq_tokens, 2):
                cooc_postings[(a, b)].add(study_id)

        # ---- Write JSONL outputs
        with open(self.token_postings_path, "w", encoding="utf-8") as f_tok:
            for tok, studies in sorted(token_postings.items()):
                entry = {"token": tok, "df": len(studies), "studies": sorted(studies)}
                f_tok.write(json.dumps(entry) + "\n")

        with open(self.relation_postings_path, "w", encoding="utf-8") as f_rel:
            for (src, rel, dst), studies in sorted(relation_postings.items()):
                entry = {"src": src, "rel": rel, "dst": dst, "df": len(studies), "studies": sorted(studies)}
                f_rel.write(json.dumps(entry) + "\n")

        with open(self.cooc_postings_path, "w", encoding="utf-8") as f_co:
            for (u, v), studies in sorted(cooc_postings.items()):
                entry = {"u": u, "v": v, "df": len(studies), "studies": sorted(studies)}
                f_co.write(json.dumps(entry) + "\n")

        print("Done. Postings saved:")
        print(f"- {self.token_postings_path}")
        print(f"- {self.relation_postings_path}")
        print(f"- {self.cooc_postings_path}")

    # -----------------------------------------------------------------------
    # Step 3: Build global graphs (NetworkX) from local subgraphs
    # -----------------------------------------------------------------------
    def build_global_graph_from_subgraphs(self):
        """
        Build global graphs from saved local GraphMLs:
        - Nodes = unique token_lower strings (stored by integer IDs; 'token' attr keeps string).
        - G_rel: directed multigraph for typed relations (weights = corpus counts).
        - G_cooc: undirected graph for co-occurrence (weights = doc frequency of pair).
        Saves to self.global_graph_dir.
        """
        print(f"Building global graphs from local subgraphs (NetworkX) into: {self.global_graph_dir}")

        token_to_id = {}
        id_to_token = {}
        next_id = 0

        relation_counts = Counter()  # (src_id, dst_id, rel_type) -> count
        cooc_counts = Counter()  # (min_id, max_id) -> count

        def get_token_id(token_str: str) -> int:
            nonlocal next_id
            t = (token_str or "").lower()
            if t not in token_to_id:
                token_to_id[t] = next_id
                id_to_token[next_id] = t
                next_id += 1
            return token_to_id[t]

        # Walk local subgraphs
        local_files = [f for f in os.listdir(self.local_graph_dir) if f.endswith(".graphml")]
        for filename in tqdm(local_files):
            subgraph_path = os.path.join(self.local_graph_dir, filename)
            try:
                local_graph = nx.read_graphml(subgraph_path)
            except Exception as e:
                print(f"Warning: failed to read {subgraph_path}: {e}")
                continue

            # Collect tokens in this study (use set to avoid duplicate pairs)
            tokens_this_study_ids = []

            # Nodes
            for node_key, node_attrs in local_graph.nodes(data=True):
                token_lower = node_attrs.get("token_lower", "")
                tid = get_token_id(str(token_lower))
                tokens_this_study_ids.append(tid)

            # Edges (relations)
            for u, v, edge_attrs in local_graph.edges(data=True):
                src_token = local_graph.nodes[u].get("token_lower", "")
                dst_token = local_graph.nodes[v].get("token_lower", "")
                rel_type = edge_attrs.get("type", "")
                sid = get_token_id(str(src_token))
                did = get_token_id(str(dst_token))
                relation_counts[(sid, did, str(rel_type))] += 1

            # Co-occurrence (within-study unique pairs)
            uniq = sorted(set(tokens_this_study_ids))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    cooc_counts[(uniq[i], uniq[j])] += 1

        # Build global graphs
        # Co-occurrence (undirected, weighted)
        G_cooc = nx.Graph()
        G_cooc.add_nodes_from(range(next_id))
        nx.set_node_attributes(G_cooc, {i: id_to_token[i] for i in range(next_id)}, name="token")
        G_cooc.add_weighted_edges_from((u, v, w) for (u, v), w in cooc_counts.items())

        # Relation multigraph (directed, per-type edges with counts)
        G_rel = nx.MultiDiGraph()
        G_rel.add_nodes_from(range(next_id))
        nx.set_node_attributes(G_rel, {i: id_to_token[i] for i in range(next_id)}, name="token")
        for (u, v, rel), w in relation_counts.items():
            G_rel.add_edge(u, v, key=rel, type=rel, weight=int(w))

        # Save to the global graph directory
        nx.write_graphml(G_cooc, self.global_cooc_path)
        nx.write_graphml(G_rel, self.global_rel_path)

        print("Done. Global graphs saved:")
        print(f"- {self.global_cooc_path}")
        print(f"- {self.global_rel_path}")

