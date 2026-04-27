# Newton Examples

Hydroelastic contact demos built on [Newton](https://github.com/newton-physics/newton) — NVIDIA's GPU-accelerated rigid-body physics engine.

---

## RJ45 Hydroelastic Contact Demo

An RJ45 network plug being inserted into a socket, showcasing Newton's **hydroelastic contact model**.

Standard contact solvers produce a single point (or a handful of points) per contact pair — no patch shape, no area information.  Hydroelastic contacts run a marching-cubes SDF intersection between the two bodies and produce a full triangulated contact patch.  This demo visualises that patch in real time with a pressure-coloured iso-surface.

### Value proposition

| | Standard contacts | Hydroelastic contacts |
|---|---|---|
| Contact representation | ~5 points | ~23 000 triangulated faces |
| Solver input | ~5 points | ~205 reduced contacts |
| Contact area | not available | live estimate |
| Pressure distribution | not available | full iso-surface |
| **Contact reduction** | — | **~99 %** (hydro faces → solver) |

### Demo controls

| Input | Action |
|---|---|
| Gizmo arrows | Drag plug into socket along Y-axis |
| Right-drag | Spring-pick any body |
| `H` | Toggle ImGui side-panel |
| `Space` | Pause / resume |
| `F` | Frame camera on model |

The ImGui panel exposes live metrics and three visualisation sliders:

- **Opacity** — dim the patch to see the geometry behind it
- **Density (step)** — subsample for performance (1 = every face, cleanest)
- **Gamma** — adjust colour contrast in the pressure distribution

---

## File layout

```
newton-examples/
├── rj45_hydro.py          # Scene construction, physics kernels, simulation loop
├── hydro_contact_viz.py   # Reusable HydroContactViz class (ImGui + patch rendering)
├── tests/
│   ├── test_hydro_contact_viz.py   # Unit tests for colormap, pressure polarity, metrics
│   └── test_constants.py           # Sanity checks for physics constants
└── README.md
```

`hydro_contact_viz.py` is designed to be **reusable** — drop `HydroContactViz` into any Newton example that uses a `CollisionPipeline` with `output_contact_surface=True`.

---

## Prerequisites

### System

- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (Turing or newer)
- **CUDA Toolkit**: 12.x
- **Driver**: ≥ 525.60 (Windows) / ≥ 520.61 (Linux)

### Python

- Python 3.10 or newer

### Python packages

```
newton
warp-lang
usd-core
numpy
```

Install everything at once:

```powershell
pip install newton warp-lang usd-core numpy
```

> **Note:** `newton` pulls in `warp-lang` automatically. If you have a CUDA-capable GPU, Warp will use it; otherwise it falls back to CPU (significantly slower for this demo).

---

## Running the demos

Clone or download this repository, then open a terminal in the project folder.

### Hydroelastic demo (this repo)

```powershell
cd newton-examples
python rj45_hydro.py
```

### Vanilla Newton RJ45 demo (for comparison)

The original Newton example uses standard point contacts with no patch visualisation:

```powershell
python -m newton.examples contacts_rj45_plug
```

---

## Running the tests

The test suite covers pure-Python logic (colormap, pressure polarity, area computation, constants) and does **not** require a GPU or a Newton installation.

```powershell
cd newton-examples
pip install pytest
pytest tests/ -v
```

Expected output:

```
tests/test_constants.py::TestShapeConstants::test_shape_kh_positive         PASSED
tests/test_constants.py::TestShapeConstants::test_shape_kh_reasonable_range PASSED
...
tests/test_hydro_contact_viz.py::TestJetColormap::test_blue_at_zero          PASSED
tests/test_hydro_contact_viz.py::TestJetColormap::test_red_at_one            PASSED
tests/test_hydro_contact_viz.py::TestPressurePolarity::test_high_pressure_is_red  PASSED
...
```

---

## How it works

### Hydroelastic pipeline

```python
from newton.geometry import HydroelasticSDF

sdf_hydro_cfg = HydroelasticSDF.Config(
    output_contact_surface=True,  # generate triangulated patch
    reduce_contacts=True,         # reduce ~23 K faces to ~205 solver contacts
)
pipeline = newton.CollisionPipeline(
    model,
    reduce_contacts=True,
    broad_phase="explicit",
    sdf_hydroelastic_config=sdf_hydro_cfg,
)
```

Shapes are marked hydroelastic via `ShapeConfig(is_hydroelastic=True, kh=5e7)`.

### Visualiser

```python
from hydro_contact_viz import HydroContactViz

viz = HydroContactViz(collision_pipeline, model, viewer)

# Each frame, after collide():
viz.update(contacts)   # GPU→CPU sync (throttled to every 5 frames)

# Inside render():
viz.render()           # emits pressure-coloured line segments
```

The visualiser reads `contact_surface_depth` per triangle and maps it to colour using an **inverted** jet colormap: triangles closest to the pressure centre have the smallest SDF magnitude and map to **red**; triangles at the patch edge map to **blue**.

---

## License

Apache 2.0 — see individual file headers.
