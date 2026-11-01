# AI Tool Comparison

**Task:** Gauss-Newton optimisation with Levenberg-Marquardt damping, pure NumPy, on a non-linear least-squares problem (Track A).

**Tools compared:** Antigravity (Google DeepMind) and Claude Code (claude-sonnet-5-5, Anthropic).

## Comparative assessment

| Dimension | Antigravity | Claude Code (claude-sonnet-5-5) |
|---|---|---|
| Model / architecture | Agentic coding engine | Agentic coding assistant (Claude Sonnet 5.5) |
| Formulation accuracy | Correct Jacobian J, normal equations (JᵀJ + λI)Δθ = Jᵀr and residual vector r(θ); did not confuse the Gauss-Newton matrix with the full Hessian | Correct J and normal equations for min ½‖r(p)‖²; fit of y = a·exp(bx) recovered (2.001, 1.499) against true (2, 1.5) |
| Hallucination / error rate | No library hallucination; stayed with vectorised NumPy, no SciPy or PyTorch wrappers | No errors in the Gauss-Newton code; pure NumPy. Errors elsewhere in the session are in `FAILURE_LOG.md` |
| Edge-case handling | Detected singular / near-singular JᵀJ and added adaptive Levenberg-Marquardt damping λ | Fixed damping of 1e-9 only; no adaptive λ and no accept/reject step; fixed 30 iterations with no convergence test |
| Prompt iterations | 1 iteration for a complete, numerically stable optimiser class with unit tests | 1 iteration (as part of a larger request); no separate unit test for Gauss-Newton |
| Generation speed | Direct response with streamed tool actions and immediate file writes | Direct, with tool calls writing files and running the code |
| Explainability for viva | High: docstrings cite textbook equations, step-by-step comments | Moderate: concise inline comments |

## Observations

**Antigravity**
- *Strengths:* Generated self-contained Python using pure NumPy, including automated numerical tests comparing the analytic Jacobian steps against finite differences. Immediately created the corresponding test suite and formatted outputs.
- *Limitations / points for human review:* Needs explicit constraint guidance to prevent over-engineering helper abstractions.

**Claude Code**
- *Strengths:* Compact, correct Gauss-Newton routine in pure NumPy that converged to the true parameters on the test problem (2.001, 1.499 against 2, 1.5); fast to produce and easy to read.
- *Limitations / points for human review:* Simpler than the Levenberg-Marquardt specification: fixed damping, no adaptive λ or trust-region accept/reject step, fixed iteration count with no convergence test, and no finite-difference check of the Jacobian or separate unit test.

**Conclusion.** For this task Antigravity gave the more complete and robust optimiser. Claude Code gave the shorter implementation, and its output was correct on the test problem.

*Basis: the Claude Code column describes its Track A implementation in this repository (`tracks/track_a.py`); the two tools were not given a byte-identical prompt.*
