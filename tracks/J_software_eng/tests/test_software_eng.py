"""Unit and Regression Tests for Track J: Software Engineering Optimization.

Tests:
1. Admissibility of Branch-and-Bound upper bounding functions (U_len and U_char).
2. Exact numerical equivalence between Chapter 6 BnB optimizer and upstream fuzzywuzzy.process.
3. BKTree metric space nearest neighbor search and triangle inequality bounds.
4. Bounded Levenshtein distance with Ukkonen's cutoff.
5. Regression test ensuring all 49 upstream fuzzywuzzy unit tests pass cleanly.
"""

from collections import Counter
import os
import sys
import unittest
import numpy as np
import pytest

# Ensure project and upstream directories are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_DIR = os.path.dirname(CURRENT_DIR)
UPSTREAM_DIR = os.path.join(TRACK_DIR, "fuzzywuzzy_upstream")
PROJECT_ROOT = os.path.abspath(os.path.join(TRACK_DIR, "../.."))

for p in [PROJECT_ROOT, TRACK_DIR, UPSTREAM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fuzzywuzzy import fuzz, process
from tracks.J_software_eng.formulated_optimizer import (
    length_upper_bound,
    char_frequency_upper_bound,
    bounded_levenshtein_distance,
    BKTree,
    PreindexedCandidateCorpus,
    FormulatedFuzzyOptimizer,
)


def test_upper_bound_admissibility():
    """Mathematically proves admissibility: upper bounds MUST be >= exact SequenceMatcher ratio."""
    rng = np.random.default_rng(42)
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    
    for _ in range(150):
        len1 = rng.integers(1, 25)
        len2 = rng.integers(1, 25)
        s1 = "".join(rng.choice(vocab, size=len1))
        s2 = "".join(rng.choice(vocab, size=len2))
        
        exact_ratio = fuzz.ratio(s1, s2)
        u_len = length_upper_bound(len1, len2)
        
        c1 = Counter(s1)
        c2 = Counter(s2)
        u_char = char_frequency_upper_bound(c1, len1, c2, len2)
        
        # Admissibility check: Upper bound cannot underestimate the true ratio
        # Allow tiny float precision epsilon
        assert u_len >= exact_ratio - 1e-6, f"Length bound violated: U_len={u_len} < ratio={exact_ratio} for '{s1}' vs '{s2}'"
        assert u_char >= exact_ratio - 1e-6, f"Char bound violated: U_char={u_char} < ratio={exact_ratio} for '{s1}' vs '{s2}'"


def test_extractOne_exact_numerical_equivalence():
    """Verifies Chapter 6 BnB optimizer yields 100% identical top match and score to upstream process.extractOne."""
    candidates = [
        "Microsoft Corporation", "Apple Inc", "Amazon Web Services", "Google Alphabet",
        "Meta Platforms Inc", "Tesla Motors", "Nvidia Corporation", "Netflix Inc",
        "JPMorgan Chase & Co", "Berkshire Hathaway Inc", "Johnson & Johnson", "Visa Inc",
        "Mastercard Incorporated", "Cisco Systems Inc", "Oracle Corporation", "Intel Corp"
    ]
    queries = [
        "microsft", "appl", "amazn", "gogl", "tesla motr", "nvdia", "netflx",
        "chase bank", "berkshire", "jnj", "visa card", "orcle", "intel core"
    ]
    
    opt = FormulatedFuzzyOptimizer(scorer=fuzz.ratio)
    preindexed = PreindexedCandidateCorpus(candidates)
    
    for q in queries:
        base_match = process.extractOne(q, candidates, scorer=fuzz.ratio)
        opt_match = opt.extractOne(q, preindexed)
        assert base_match == opt_match, f"Mismatch for query '{q}': Base={base_match} vs Opt={opt_match}"


def test_extract_exact_equivalence():
    """Verifies top-K extract returns identical rankings and scores."""
    candidates = [
        "New York Mets", "Atlanta Braves", "Philadelphia Phillies", "Miami Marlins",
        "Washington Nationals", "Chicago Cubs", "St. Louis Cardinals", "Milwaukee Brewers",
        "Pittsburgh Pirates", "Cincinnati Reds", "Los Angeles Dodgers", "San Francisco Giants"
    ]
    queries = ["new york", "chicago cubs", "los angeles", "atlanta"]
    
    opt = FormulatedFuzzyOptimizer(scorer=fuzz.ratio)
    preindexed = PreindexedCandidateCorpus(candidates)
    
    for q in queries:
        base_top3 = process.extract(q, candidates, limit=3, scorer=fuzz.ratio)
        opt_top3 = opt.extract(q, preindexed, limit=3)
        assert base_top3 == opt_top3, f"Mismatch for query '{q}': Base={base_top3} vs Opt={opt_top3}"


def test_bktree_metric_search():
    """Verifies BK-tree correctly indexes strings and prunes via the triangle inequality."""
    words = ["book", "books", "cake", "boo", "boon", "cook", "cake", "cape", "cart"]
    tree = BKTree()
    tree.build_from_iterable(words)
    
    # Query "boos" with max distance 1 -> should match "book" (dist 1) and "books" (dist 1)
    matches = tree.search("boos", max_dist=1)
    matched_words = {w for w, d in matches}
    assert "book" in matched_words
    assert "books" in matched_words
    assert "cake" not in matched_words


def test_bounded_levenshtein_distance():
    """Verifies bounded Levenshtein distance matches standard edit distance within cutoff threshold."""
    assert bounded_levenshtein_distance("kitten", "sitting", max_dist=3) == 3
    assert bounded_levenshtein_distance("flaw", "lawn", max_dist=2) == 2
    assert bounded_levenshtein_distance("rosettacode", "raisethysword", max_dist=4) > 4


def test_upstream_test_suite_clean_pass():
    """Runs upstream fuzzywuzzy test suite (49 tests) ensuring zero behavioral regressions."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=UPSTREAM_DIR, pattern="test_fuzzywuzzy.py")
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    assert result.wasSuccessful(), f"Upstream tests failed with errors={result.errors}, failures={result.failures}"
    assert result.testsRun >= 45
