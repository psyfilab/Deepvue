"""Feature extraction helpers for text classifiers."""

from __future__ import annotations

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

logger = logging.getLogger(__name__)


def build_feature_pipeline(
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 4),
    word_max_features: int = 20_000,
    char_max_features: int = 30_000,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """Build the shared TF-IDF feature union used by all learned models."""

    logger.debug(
        "Building feature pipeline with word ngrams=%s, char ngrams=%s",
        word_ngram_range,
        char_ngram_range,
    )

    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngram_range,
                    min_df=1,
                    sublinear_tf=sublinear_tf,
                    max_features=word_max_features,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=1,
                    sublinear_tf=sublinear_tf,
                    max_features=char_max_features,
                ),
            ),
        ]
    )
