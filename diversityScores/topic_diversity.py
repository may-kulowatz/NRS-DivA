import os


def _parse_user_articles(user_articles_file):
    users = {}
    with open(user_articles_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_id   = parts[0]
            ids       = parts[1][1:-1].split(",") if parts[1][1:-1] else []
            topics    = parts[2][1:-1].split(",") if parts[2][1:-1] else []
            subtopics = parts[3][1:-1].split(",") if parts[3][1:-1] else []
            users[user_id] = (ids, topics, subtopics)
    return users


def topic_diversity(user_articles_file):
    """Average of (unique topics / total topic assignments) across users with more than one click.

    An article may carry several topics. They are stored as a single
    "|"-separated group in the topics field (e.g. "Crime|Violent_crime"); all of
    an article's topics are flattened and counted. For single-topic datasets
    (e.g. MIND) each group is just one topic, so this reduces to the plain
    unique-topics / number-of-articles ratio.

    Articles with no topics are stored as the sentinel "none"; these are
    filtered out and contribute to neither the unique count nor the total. A
    user left with no topics after filtering is skipped entirely.
    """
    per_user = []
    for _, (ids, topics, _) in _parse_user_articles(user_articles_file).items():
        if len(ids) <= 1:
            continue
        flat = [t for group in topics for t in group.split("|") if t != "none"]
        if not flat:
            continue
        per_user.append(len(set(flat)) / len(flat))
    return sum(per_user) / len(per_user) if per_user else 0.0


def subtopic_diversity(user_articles_file, category="news"):
    """Average of (unique subtopics / total clicks in category) across users with more than one click.
    Users who have no clicks in the given category contribute 0.

    This metric only makes sense when subtopics are nested under a parent
    category (as in MIND). Datasets whose subcategories cannot be mapped to a
    parent category (e.g. eb-nerd) should not call this — the pipeline skips it
    for them.
    """
    per_user = []
    for _, (ids, topics, subtopics) in _parse_user_articles(user_articles_file).items():
        if len(ids) <= 1:
            continue
        filtered = [sub for top, sub in zip(topics, subtopics) if top == category]
        per_user.append(len(set(filtered)) / len(filtered) if filtered else 0.0)
    return sum(per_user) / len(per_user) if per_user else 0.0


if __name__ == "__main__":
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mind_dir = os.path.join(_project_dir, "data", "MIND")

    files = {
        "random":       os.path.join(mind_dir, "user_articles_random.txt"),
        "popular":      os.path.join(mind_dir, "user_articles_popular.txt"),
        "ground_truth": os.path.join(mind_dir, "user_articles_ground_truth.txt"),
    }

    for name, path in files.items():
        print(f"\n=== {name} ===")
        print(f"  Topic diversity:              {topic_diversity(path):.4f}")
        print(f"  Subtopic diversity (news):    {subtopic_diversity(path, category='news'):.4f}")