"""mind_news dataset adapter.

mind_news is a news-only subset of MIND (built by mind_news/prepare.py): every
article sits in the MIND ``news`` category, so the category column carries no
variety at all. What *does* vary is the subcategory, so this adapter promotes the
subcategory into the topic slot — the diversity metrics then measure variety
among news subcategories (what the old "subtopic diversity" used to capture).

Impressions and titles are parsed exactly as for MIND (the files are in the same
format); only ``load_article_meta`` differs.
"""

from dataset_module.mind import adapter as mind_adapter
from dataset_module.mind.adapter import load_impressions, load_titles

__all__ = ["load_impressions", "load_article_meta", "load_titles"]


def load_article_meta(news_file):
    """Read news.tsv into ``{news_id: (subtopic, "none")}``.

    The subcategory is promoted into the topic slot (index 0) so ``topic_diversity``
    measures news-subcategory variety; the subtopic slot is left as the unused
    ``"none"`` sentinel.
    """
    return {
        aid: (sub, "none")
        for aid, (_category, sub) in mind_adapter.load_article_meta(news_file).items()
    }