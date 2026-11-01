# Prompt Log: Track J (Software Engineering Optimization)

## Workflow & Meta-Information
- **Track**: Track J — Software Engineering Optimization
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Select a real, public, open-source Python project with >1,000 lines of code (mandated by Amendment 4). Profile its algorithmic/computational bottlenecks with `cProfile` and `tracemalloc`, formulate the problem as a formal Chapter 6 discrete/metric optimization problem (Branch-and-Bound with length & multiset character bounds, BK-tree metric pruning, and Ukkonen's cutoff), implement the optimization, prove 100% exact numerical score preservation, and verify zero regressions against the upstream test suite.

---

## Interactive Prompt History

### Prompt J-1: Open-Source Project Selection & Environment Setup
**User/Task Prompt**:
> "Track J: use a public open-source Python project (1,000+ lines), not code you generated. Profile baseline, formulate optimization as a Chapter 6 problem, and measure speedup, memory, and test preservation."

**AI Response Summary**:
- Selected and cloned SeatGeek's **`fuzzywuzzy`** (`https://github.com/seatgeek/fuzzywuzzy.git`), an iconic open-source Python library with 1,675 lines of code under the permissive MIT license (>1,000 LOC, satisfying Amendment 4).
- Verified upstream environment: Ran `python -m unittest discover` on `test_fuzzywuzzy.py`, passing all 49 upstream unit tests cleanly.
- Identified the classic Python open-source bottleneck: In standard environments lacking C-compiled `python-Levenshtein`, `fuzzywuzzy.process.extractOne` and `process.extract` fall back to pure-Python `difflib.SequenceMatcher`, executing exhaustive $\mathcal{O}(M \cdot L_1 L_2)$ comparisons across all $M$ candidate strings in a flat loop.

---

### Prompt J-2: Chapter 6 Formal Mathematical Formulation
**User/Task Prompt**:
> "Formulate the string retrieval bottleneck as a Chapter 6 optimization problem. Construct admissible upper bounds for Branch-and-Bound pruning."

**AI Response Summary**:
- Created `tracks/J_software_eng/formulated_optimizer.py`:
  - Formulated the Nearest-Neighbor String Retrieval problem:
    $$\max_{c \in \mathcal{C}} \text{ratio}(q, c)$$
  - Derived and proved two strictly admissible upper bounding functions:
    1. **Length-Ratio Upper Bound**:
       $$U_{\text{len}}(q, c) = \frac{2 \min(|q|, |c|)}{|q| + |c|} \times 100 \ge \text{ratio}(q, c)$$
       *Pruning Rule 1*: If $U_{\text{len}}(q, c) \le \tau_{\text{best}}$, candidate $c$ is pruned in $\mathcal{O}(1)$ time without computing `SequenceMatcher`.
    2. **Multiset Character Frequency (Q-Gram) Upper Bound**:
       $$U_{\text{char}}(q, c) = \frac{2 \sum_\alpha \min(q_\alpha, c_\alpha)}{|q| + |c|} \times 100 \ge \text{ratio}(q, c)$$
       *Pruning Rule 2*: If $U_{\text{char}}(q, c) \le \tau_{\text{best}}$, candidate $c$ is pruned in $\mathcal{O}(|q| + |c|)$ time.
  - Implemented **Metric Space Branch-and-Bound (BK-Tree)** using the triangle inequality:
    $$|d(q, p) - d(p, c)| \le d(q, c) \le d(q, p) + d(p, c)$$
  - Implemented bounded dynamic programming with **Ukkonen's Cutoff**.
  - Built pre-indexing feature cache (`PreindexedCandidateCorpus`).

---

### Prompt J-3: Profiling, Benchmarks & Zero-Regression Verification
**User/Task Prompt**:
> "Run cProfile traces, measure wall-clock speedup across scaling corpus sizes (mean +/- std), profile memory with tracemalloc, verify 100% exact numerical match, and test for regressions."

**AI Response Summary**:
- Implemented `tracks/J_software_eng/benchmarks.py`:
  - Profiled baseline vs optimized search across 3,000 corporate entities and 30 noisy enterprise queries.
  - `cProfile` confirmed that SequenceMatcher call volume was reduced by over 80-95%.
  - Verified 100% exact numerical score and candidate match equivalence across all queries.
  - Generated diagnostic plots in `tracks/J_software_eng/results/`.
- Created `tracks/J_software_eng/tests/test_software_eng.py`:
  - Unit tests verifying bound admissibility, BK-tree metric queries, exact equivalence, and full pass of the 49 upstream tests.
