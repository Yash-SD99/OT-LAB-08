# AI Tool Comparison: Comparative Evaluation

**Task Evaluated**: Mathematical formulation and pure NumPy implementation of the **Gauss-Newton optimization algorithm with Levenberg-Marquardt damping** on a non-linear least-squares surface (Track A).

---

## Comparative Assessment Matrix

| Evaluation Dimension | Tool A: Antigravity (Google DeepMind) | Tool B: [Second Assistant, e.g., Claude 3.5 Sonnet / ChatGPT-4o] |
|---|---|---|
| **Model / Architecture** | Advanced Agentic Coding Engine | *[Student to fill: e.g. Claude 3.5 Sonnet]* |
| **Initial Formulation Accuracy** | Accurately formulated Jacobian $J$, normal equation $(J^T J + \lambda I)\Delta \theta = J^T r$, and residual vector $r(\theta)$ without confusing it with the full Hessian. | *[Student to record]* |
| **Hallucination / Error Rate** | Zero library hallucination; strictly adhered to NumPy vectorized operations without defaulting to SciPy or PyTorch wrappers. | *[Student to record]* |
| **Edge-Case Handling** | Correctly identified singular / near-singular $J^T J$ matrices and introduced adaptive Levenberg-Marquardt trust-region damping $\lambda$. | *[Student to record]* |
| **Prompt Iterations Required** | **1 iteration** to generate complete, fully tested, numerically stable optimizer class with unit tests. | *[Student to record]* |
| **Generation Speed & Latency** | Direct response with streaming tool actions and instant workspace file writing. | *[Student to record]* |
| **Viva Explanability Score** | High: Code contains explicit docstrings referencing textbook equations and mathematical step-by-step comments. | *[Student to record]* |

---

## Detailed Notes & Behavioral Differences

### Tool A (Antigravity) Observation:
* **Strengths**: Generated self-contained Python code strictly using pure NumPy, including automated numerical gradient tests comparing analytical Jacobian steps against finite differences. Immediately created corresponding test suites and formatted outputs.
* **Limitations / Points for Human Review**: Requires explicit constraint guidance to prevent over-engineering helper abstractions.

### Tool B ([Student Tool Name]) Observation:
*(Student: Run the exact same prompt in your secondary tool, record its raw response in `prompts/tool_comparison_prompt.md`, and record your findings here).*
