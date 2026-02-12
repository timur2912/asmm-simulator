# AGENTS.md

Repository notes for future work:

- Package module: `asmm_simulator` (installed from `src/`).
- Main CLI: `python -m asmm_simulator ...` (entrypoint in `asmm_simulator/__main__.py`).
- Reproduction outputs are written to `outputs/` by default.
- Simulation uses vectorized (per-time-step) updates across paths with common random numbers.
- Default stability choices: allow quote crossing in prices (`allow_cross=True`) while treating deltas as distances for intensity (`floor_intensity_delta=True`) and cap Bernoulli probabilities (`cap_prob=True`).
