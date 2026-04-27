# Newton Examples

A collection of demos and extensions built on [Newton](https://github.com/newton-physics/newton) — NVIDIA's open-source GPU-accelerated rigid-body physics engine.

Each example lives in its own subfolder under `examples/` with its own code, assets, and tests.

---

## Prerequisites

### System

- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (Turing or newer)
- **CUDA Toolkit**: 12.x
- **Driver**: ≥ 525.60 (Windows) / ≥ 520.61 (Linux)
- **Python**: 3.10 or newer

### Install Newton

```powershell
pip install newton warp-lang usd-core numpy
```

---

## Examples

| Example | Description |
|---|---|
| [rj45_hydro](examples/rj45_hydro/) | RJ45 plug insertion with hydroelastic contacts and live pressure patch visualisation |

---

## Running tests

Each example has its own test suite. To run all tests across every example:

```powershell
python -m pytest examples/ -v
```

Or for a specific example:

```powershell
python -m pytest examples/rj45_hydro/tests/ -v
```

---

## Contributing a new example

1. Create a subfolder: `examples/<your_example>/`
2. Add your Python files and a `tests/` directory
3. Include a `README.md` inside the subfolder describing what the example demonstrates and how to run it

---

## License

Apache 2.0 — see individual file headers.
