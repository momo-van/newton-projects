# Newton Examples — Codex Context

## Repo

`C:\Vibe Coding\newton-examples` — a collection of Newton physics engine demos built with Warp GPU kernels.

Owner: mmohajerani@nvidia.com (NVIDIA)

---

## Stack

| Layer | Library |
|---|---|
| Physics engine | [Newton](https://github.com/newton-physics/newton) |
| GPU kernels | [Warp](https://github.com/NVIDIA/warp) (`warp-lang`) |
| 3-D geometry | trimesh + Shapely (2-D profiles → extruded meshes) |
| Scene format | USD (`usd-core`, `pxr`) — Z-up, metres |
| Visualization | Rerun (`rerun-sdk`) + Newton built-in viewer (ImGui) |
| Tests | pytest — **stub-based, no GPU/Newton required** |

Install: `pip install newton warp-lang usd-core numpy trimesh shapely rerun-sdk`

---

## Repo layout

```
examples/
  rj45_hydro/          # RJ45 hydroelastic contact demo (stable)
  spline_insertion/    # DIN 5480 shaft→hub insertion (active work)
    gen_geometry.py    # generates shaft.usd + hub.usd (16-tooth default)
    gen_6groove.py     # alternative 6-tooth z=6 variant (4 mm module)
    sim_insertion.py   # main simulation
    compare_teeth.py   # Rerun side-by-side z=6 | z=8 | z=16 comparison
    view_geometry.py   # static geometry viewer
    view_usd_rerun.py  # USD viewer via Rerun
    convert_sourced.py # converts sourced STL files to USD
    tests/
      test_sim_insertion.py  # unit tests (stub-based)
```

---

## Multi-phase project vision

The spline_insertion example is being redesigned as a 5-phase demo of Newton capabilities:

| Phase | Name | Key Newton feature |
|---|---|---|
| 1 | Geometry | Parametric DIN 5480 mesh generation, USD export |
| 2 | Kinematic | Rotation + translation drive, SDF + hydro contacts |
| 3 | Dynamics | VBD solver, virtual manipulation force/torque |
| 4 | Thermal | Custom Warp heat diffusion + friction heating (first principles) |
| 5 | Surrogate | PhysicsNeMo model inference injected per-phase |

**Design decisions (all answered):**
- Q1: Straight-key / parallel profile — matches reference video ✅
- Q2: Phase 2 = INTERACTIVE — keyboard/slider controls for rotation speed + axial push
- Q3: Phase 4 = SURFACE temperature on contact patches; Q = μ·N·v from Phase 3 forces
- Q4: Phase 5 = STUB interface — no checkpoint yet; plug-in point for contact force surrogate

**Phase 2 hydro contact viz (adapted from rj45_hydro):**
- Original used Newton ViewerGL log_lines + ImGui
- Phase 2 port: rr.Mesh3D colored triangles (pressure), rr.Arrows3D (forces), rr.Scalars timeseries
- Jet colormap preserved: red = high pressure, blue = low

**Planned folder structure:**
```
examples/spline_insertion/
  phase1_geometry/
  phase2_kinematic/
  phase3_dynamics/
  phase4_thermal/
  phase5_surrogate/
```

---

## Real insertion mechanics (from reference video)

The insertion is a **search-and-engage** process, NOT pure axial translation:

1. **Approach** — shaft lowered to hover just above hub bore entrance
2. **Angular search** — operator wraps fingers around shaft, applies light axial load + slow rotation; teeth ride on land faces → high contact force
3. **Snap-to-engage** — once a tooth-gap alignment is found, axial resistance drops sharply, shaft drops under gravity/push
4. **Press-fit seat** — operator pushes straight down to full depth

For z=16 there are 16 valid engagement angles (every 22.5°). SDF contact forces naturally encode
"teeth riding on lands" vs "teeth dropping into grooves" — this is the core Newton demo moment.

Phase 2 kinematic driver must combine **rotational sweep + axial force/displacement**, not just Z-translation.

---

## Current state (pre-refactor)

**Geometry** (DIN 5480 involute spline, 30° pressure angle):
- Default: Z=16, module=2 mm → 32 mm pitch dia, 40 mm length, 52 mm hub OD
- 6-tooth variant: Z=6, module=4 mm → ~32 mm tip dia, 40 mm length, 64 mm hub OD
- `gen_geometry.py` → `shaft.usd` + `hub.usd`; also exports `.glb`

**Simulation** (`sim_insertion.py`):
- Shaft: zero-mass kinematic body; position/velocity overridden every substep by `_drive_shaft` Warp kernel
- Hub: fixed to world (body index -1)
- Contacts: Newton mesh-SDF (`CollisionPipeline`, broad_phase="explicit")
- Solver: `SolverVBD` (12 iterations, 60 fps, 8 substeps/frame)
- Insertion speed: 5 mm/s; shaft starts 30 mm above hub, ends at hub centre
- ImGui side panel: status, shaft Z, progress %, depth into bore, Pause/Resume/Reset

**Tests** (`tests/test_sim_insertion.py`):
- All warp/newton/pxr imports replaced with lightweight stubs
- Tests cover: constants, insertion advancement, pause behaviour, reset, progress/depth math, timing, `_load_mesh` return shape
- Run: `pytest examples/spline_insertion/tests/ -v`

---

## Conventions

- All units SI (metres, radians, seconds)
- USD: Z-up axis, `metersPerUnit=1.0`
- Warp kernels: `@wp.kernel`, launched with `wp.launch(..., dim=1, ...)`
- Body transforms: `wp.transform` (position, quaternion)
- Spatial velocity: `wp.spatial_vector(ω_x, ω_y, ω_z, v_x, v_y, v_z)`
- Test stubs live entirely inside the test file — no separate conftest stubs for spline_insertion

---

## Running

```powershell
# Generate geometry first (one-time)
cd examples\spline_insertion
python gen_geometry.py        # 16-tooth default
python gen_6groove.py         # 6-tooth variant (overwrites shaft.usd/hub.usd)

# Run simulation
python sim_insertion.py

# Compare tooth counts in Rerun
python compare_teeth.py

# Tests (no GPU needed)
python -m pytest examples\ -v
```
