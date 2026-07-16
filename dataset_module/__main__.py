"""Prepare a dataset's raw inputs (download / build them).

Run with ``python -m dataset_module [dataset]`` to prepare one dataset, or with no
argument (or ``--all``) to prepare every dataset. A single dataset can also be
prepared through its preparation file, e.g. ``python -m dataset_module.mind.prepare``.

PRE  : network access for datasets with a known download source (MIND, and
       mind_news which derives from it). EB-NeRD has no public URL — its files
       must already be on disk; preparation only verifies they are present.
POST : each prepared dataset's raw inputs (and optional content-diversity utils
       bundle) exist under ``data/datasets/<dataset>/``.
"""

import argparse

from dataset_module.common import default_input_dir
from dataset_module.ebnerd import prepare as ebnerd_prepare
from dataset_module.mind import prepare as mind_prepare
from dataset_module.mind_news import prepare as mind_news_prepare

# Order matters: mind must be prepared before mind_news, which derives from it.
_PREPARERS = (mind_prepare, ebnerd_prepare, mind_news_prepare)
_BY_DIR = {prep.DIR: prep for prep in _PREPARERS}


def _prepare(prep):
    in_dir = default_input_dir(prep.DIR)
    print(f"=== Preparing {prep.DIR} ({in_dir}) ===")
    try:
        prep.ensure_raw_data(in_dir)
        prep.ensure_utils(in_dir)
    except Exception as exc:
        print(f"  could not prepare {prep.DIR} "
              f"({exc.__class__.__name__}: {exc}); skipping it.")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m dataset_module",
        description="Prepare (download / build) datasets' raw inputs.",
    )
    p.add_argument("dataset", nargs="?", default=None,
                   help=f"dataset to prepare (one of {list(_BY_DIR)}, any case; "
                        "default: all of them)")
    p.add_argument("--all", action="store_true",
                   help="prepare every dataset (the default when no dataset is named)")
    args = p.parse_args(argv)

    if args.dataset and not args.all:
        prep = _BY_DIR.get(args.dataset.lower())
        if prep is None:
            p.error(f"unknown dataset {args.dataset!r}; choose from {list(_BY_DIR)}")
        _prepare(prep)
    else:
        for prep in _PREPARERS:
            _prepare(prep)


if __name__ == "__main__":
    main()