import os
import numpy as np
import zipfile
from collections import defaultdict
from tqdm import tqdm


def load_impressions_mind(behaviors_file):
     impressions = []
     with open(behaviors_file, encoding="utf-8") as f:
          for line in f:
               cols = line.strip().split("\t")
               impressions.append((int(cols[0]), cols[1], len(cols[4].split())))
     return impressions

def random_recommend(impressions, seed=42):
     rng = np.random.default_rng(seed)
     return [(impr_id, user_id, rng.random(n)) for impr_id, user_id, n in impressions]

def save_predictions_mind(results, output_file):
    with open(output_file, "w") as f:
        for impr_id, user_id, scores in tqdm(results):
            ranks = (np.argsort(np.argsort(scores)[::-1]) + 1).tolist()
            f.write(f"{impr_id} {user_id} [" + ",".join(map(str, ranks)) + "]\n")

    # Create zip file
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


def save_user_article_map(topk_file, output_file):
    user_articles = defaultdict(list)
    with open(topk_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_id = parts[1]
            inner_ids = parts[3][1:-1]
            if inner_ids:
                user_articles[user_id].extend(inner_ids.split(","))

    with open(output_file, "w") as f:
        for user_id, articles in user_articles.items():
            f.write(f"{user_id} [" + ",".join(articles) + "]\n")


if __name__ == "__main__":
     _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
     mind_dir = os.path.join(_project_dir, "data", "MIND")
     behaviors_file = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
     ground_truth_file = os.path.join(mind_dir, "prediction_ground_truth.txt")
     output_file = os.path.join(mind_dir, "prediction_random.txt")
     output_file_topk = os.path.join(mind_dir, "prediction_random_topk.txt")
     output_file_user_map = os.path.join(mind_dir, "user_articles_random.txt")

     impressions = load_impressions_mind(behaviors_file)
     results = random_recommend(impressions, seed=42)
     save_predictions_mind(results, output_file)
     save_predictions_mind_topk(results, behaviors_file, ground_truth_file, output_file_topk)
     save_user_article_map(output_file_topk, output_file_user_map)