"""Prepare every dataset's raw inputs in one go.

Run with ``python -m dataset_module`` to fetch/build the raw inputs and optional
utility bundles for all datasets. Each dataset can also be prepared on its own,
e.g. ``python -m dataset_module.mind.prepare``.

Note: datasets with a known download source (MIND, and mind_news which derives
from it) will fetch missing data via the Recommenders library; eb-nerd has no
public URL and only verifies its inputs are present.
"""

from dataset_module.common import default_input_dir
from dataset_module.ebnerd import prepare as ebnerd_prepare
from dataset_module.mind import prepare as mind_prepare
from dataset_module.mind_news import prepare as mind_news_prepare

# Order matters: mind must be prepared before mind_news, which derives from it.
_PREPARERS = (mind_prepare, ebnerd_prepare, mind_news_prepare)


def main():
    for prep in _PREPARERS:
        in_dir = default_input_dir(prep.DIR)
        print(f"=== Preparing {prep.DIR} ({in_dir}) ===")
        try:
            prep.ensure_raw_data(in_dir)
            prep.ensure_utils(in_dir)
        except Exception as exc:
            print(f"  could not prepare {prep.DIR} "
                  f"({exc.__class__.__name__}: {exc}); skipping it.")


if __name__ == "__main__":
    main()
