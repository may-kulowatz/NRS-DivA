import os
from collections import namedtuple


def default_input_dir(dir_name):
    """Default on-disk input directory for a dataset's raw files.

    Resolves ``<project>/data/datasets/<dir_name>`` from this file's location, so
    the per-dataset prepare modules can locate their data when run standalone
    (``python -m dataset_module``) without importing the root ``config`` (which
    imports ``dataset_module``).
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_dir, "data", "datasets", dir_name)


# Normalized, dataset-agnostic view of a single impression.
#
#   impr_id        int   — unique impression id
#   user_id        str   — user the impression was shown to
#   timestamp            — sortable impression time (popular_recommend orders by it)
#   candidate_ids  [str] — articles shown in the impression, in display order
#   labels         [int] — 1 if the matching candidate was clicked, else 0
Impression = namedtuple(
    "Impression", ["impr_id", "user_id", "timestamp", "candidate_ids", "labels"]
)
