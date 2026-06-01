import os
import numpy as np
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

def save_user_article_map(topk_file, news_file, output_file):
    topics = {}
    subtopics = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            topics[cols[0]] = cols[1]
            subtopics[cols[0]] = cols[2] if cols[2] else "none"

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
            article_topics = [topics.get(a, "unknown") for a in articles]
            article_subtopics = [subtopics.get(a, "none") for a in articles]
            f.write(f"{user_id} [" + ",".join(articles) + "] [" + ",".join(article_topics) + "] [" + ",".join(article_subtopics) + "]\n")


if __name__ == "__main__":
     _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
     mind_dir = os.path.join(_project_dir, "data", "MIND")
     behaviors_file = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
     ground_truth_file = os.path.join(mind_dir, "prediction_ground_truth.txt")
     output_file = os.path.join(mind_dir, "prediction_random.txt")
     output_file_topk = os.path.join(mind_dir, "prediction_random_topk.txt")
     news_file = os.path.join(mind_dir, "MINDsmall_dev", "news.tsv")
     output_file_user_map = os.path.join(mind_dir, "user_articles_random.txt")

     impressions = load_impressions_mind(behaviors_file)
     results = random_recommend(impressions, seed=42)
     save_predictions_mind(results, output_file)
     save_predictions_mind_topk(results, behaviors_file, ground_truth_file, output_file_topk)
     save_user_article_map(output_file_topk, news_file, output_file_user_map)