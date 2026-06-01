import os
from collections import defaultdict
from tqdm import tqdm


def load_ground_truth_mind(behaviors_file):
    results = []
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            impr_id = int(cols[0])
            user_id = cols[1]
            candidates = cols[4].split()
            labels = [int(c.split("-")[1]) for c in candidates]
            article_ids = [c.split("-")[0] for c in candidates]
            results.append((impr_id, user_id, labels, article_ids))
    return results


def save_ground_truth_mind(results, output_file):
    with open(output_file, "w") as f:
        for impr_id, user_id, labels, article_ids in tqdm(results):
            clicked = [(i + 1, article_ids[i]) for i, label in enumerate(labels) if label == 1]
            positions = [str(pos) for pos, _ in clicked]
            ids = [aid for _, aid in clicked]
            f.write(f"{impr_id} {user_id} [" + ",".join(positions) + "] [" + ",".join(ids) + "]\n")

def save_user_article_map(ground_truth_file, news_file, output_file):
    topics = {}
    subtopics = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            topics[cols[0]] = cols[1]
            subtopics[cols[0]] = cols[2] if cols[2] else "none"

    user_articles = defaultdict(list)
    with open(ground_truth_file, encoding="utf-8") as f:
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
    output_file = os.path.join(mind_dir, "prediction_ground_truth.txt")
    news_file = os.path.join(mind_dir, "MINDsmall_dev", "news.tsv")
    output_file_user_map = os.path.join(mind_dir, "user_articles_ground_truth.txt")

    results = load_ground_truth_mind(behaviors_file)
    save_ground_truth_mind(results, output_file)
    save_user_article_map(output_file, news_file, output_file_user_map)
