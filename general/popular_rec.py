import os
import numpy as np
import zipfile
from collections import defaultdict
from tqdm import tqdm


def load_impressions_mind(behaviors_file):
    rows = []
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            impr_id = int(cols[0])
            user_id = cols[1]
            timestamp = cols[2]
            history = cols[3].split() if cols[3] else []
            candidates = cols[4].split()
            rows.append((impr_id, user_id, timestamp, history, candidates))
    return rows


def popular_recommend(rows):
    # Sort by timestamp to process impressions chronologically.
    sorted_rows = sorted(rows, key=lambda r: r[2])

    click_counts = defaultdict(int)
    scores_by_id = {}

    for impr_id, user_id, timestamp, history, candidates in sorted_rows:
        article_ids = [c.split("-")[0] for c in candidates]
        scores = np.array([float(click_counts[aid]) for aid in article_ids])
        scores_by_id[impr_id] = scores

        # Update click counts with clicks observed in this impression.
        for c in candidates:
            parts = c.split("-")
            if len(parts) == 2 and parts[1] == "1":
                click_counts[parts[0]] += 1

    # Return results in original order.
    return [(impr_id, user_id, scores_by_id[impr_id]) for impr_id, user_id, _, _, _ in rows]


def save_predictions_mind(results, output_file):
    with open(output_file, "w") as f:
        for impr_id, user_id, scores in tqdm(results):
            ranks = (np.argsort(np.argsort(scores)[::-1]) + 1).tolist()
            f.write(f"{impr_id} {user_id} [" + ",".join(map(str, ranks)) + "]\n")

    zip_path = output_file.replace(".txt", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname="prediction.txt")


def save_predictions_mind_topk(results, behaviors_file, ground_truth_file, output_file):
    article_ids = {}
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            impr_id = int(cols[0])
            article_ids[impr_id] = [c.split("-")[0] for c in cols[4].split()]

    k_by_impr = {}
    with open(ground_truth_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            impr_id = int(parts[0])
            inner = parts[2][1:-1]
            k_by_impr[impr_id] = len(inner.split(",")) if inner else 0

    with open(output_file, "w") as f:
        for impr_id, user_id, scores in tqdm(results):
            k = k_by_impr.get(impr_id, 0)
            ids = article_ids[impr_id]
            top_k_idx = np.argsort(scores)[::-1][:k]
            positions = [i + 1 for i in top_k_idx]
            chosen_ids = [ids[i] for i in top_k_idx]
            f.write(f"{impr_id} {user_id} [" + ",".join(map(str, positions)) + "] [" + ",".join(chosen_ids) + "]\n")

    zip_path = output_file.replace(".txt", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname="prediction.txt")


if __name__ == "__main__":
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mind_dir = os.path.join(_project_dir, "data", "MIND")
    behaviors_file = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
    ground_truth_file = os.path.join(mind_dir, "prediction_ground_truth.txt")
    output_file = os.path.join(mind_dir, "prediction_popular.txt")
    output_file_topk = os.path.join(mind_dir, "prediction_popular_topk.txt")

    rows = load_impressions_mind(behaviors_file)
    results = popular_recommend(rows)
    save_predictions_mind(results, output_file)
    save_predictions_mind_topk(results, behaviors_file, ground_truth_file, output_file_topk)