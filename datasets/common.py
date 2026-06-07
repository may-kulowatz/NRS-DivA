from collections import namedtuple

# Normalized, dataset-agnostic view of a single impression.
#
# Every dataset adapter (datasets/mind.py, datasets/ebnerd.py) parses its own
# on-disk format into this shape, after which the recommenders and output
# writers never need to know which dataset the data came from.
#
#   impr_id        int   — unique impression id
#   user_id        str   — user the impression was shown to
#   timestamp            — sortable impression time (popular_recommend orders by it)
#   candidate_ids  [str] — articles shown in the impression, in display order
#   labels         [int] — 1 if the matching candidate was clicked, else 0
Impression = namedtuple(
    "Impression", ["impr_id", "user_id", "timestamp", "candidate_ids", "labels"]
)
