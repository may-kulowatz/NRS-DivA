"""Topic diversity from a recommender's processed per-user file.

Scores how varied a user's recommended topics (categories) are, and provides the
shared ``_parse_user_articles`` reader that ``content_diversity`` also reuses.
"""


def _parse_user_articles(processed_file):
    users = {}
    with open(processed_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_id   = parts[0]
            ids       = parts[1][1:-1].split(",") if parts[1][1:-1] else []
            topics    = parts[2][1:-1].split(",") if parts[2][1:-1] else []
            users[user_id] = (ids, topics)
    return users


def topic_diversity(processed_file):
    """Average of (unique topics / total topic assignments) across users with more than one click.

    An article may carry several topics. They are stored as a single
    "|"-separated group in the topics field; all of an article's topics are flattened and counted.

    Articles with no topics are stored as the sentinel "none"; these are
    filtered out and contribute to neither the unique count nor the total. A
    user left with no topics after filtering is skipped entirely.
    """
    per_user = []
    for _, (ids, topics) in _parse_user_articles(processed_file).items():
        if len(ids) <= 1:
            continue
        flat = [t for group in topics for t in group.split("|") if t != "none"]
        if not flat:
            continue
        per_user.append(len(set(flat)) / len(flat))
    return sum(per_user) / len(per_user) if per_user else 0.0