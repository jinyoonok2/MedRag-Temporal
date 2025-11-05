import os
import json
from tqdm import tqdm
from radgraph import RadGraph


def generate_radgraph_annotations(data_loader):
    """
    Generates RadGraph annotations by saving only JSON entity graphs,
    skipping studies that already have a JSON file.

    Args:
        data_loader: DataLoader yielding dicts with 'report_txt', 'graph_json_path'
    """
    print(f"🔍 Generating RadGraph JSON annotations (skipping existing files)")
    radgraph = RadGraph(model_type="modern-radgraph-xl")

    for batch in tqdm(data_loader, desc="Generating annotations"):
        report_texts = batch["refined_report_txt"]
        graph_json_paths = batch["graph_json_path"]

        inputs_to_run = []
        paths_to_save = []
        index_map = []

        # Filter out already existing JSONs
        for i, path in enumerate(graph_json_paths):
            if os.path.exists(path):
                continue
            inputs_to_run.append(report_texts[i])
            paths_to_save.append(path)
            index_map.append(i)

        # Skip batch if all files exist
        if not inputs_to_run:
            continue

        # Run RadGraph model on filtered inputs
        annotations_batch = radgraph(inputs_to_run)

        for i, original_index in enumerate(index_map):
            annotations = annotations_batch[str(i)]  # i is the new index after filtering
            path = paths_to_save[i]

            with open(path, "w", encoding="utf-8") as f_json:
                json.dump(annotations, f_json, indent=2)

    print("✅ Done generating RadGraph JSON annotations.")
