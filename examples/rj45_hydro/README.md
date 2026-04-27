# RJ45 Hydroelastic Contact Demo

An RJ45 network plug being inserted into a socket, showcasing Newton's **hydroelastic contact model**.

Standard solvers produce a handful of discrete contact points — no patch shape, no area. Hydroelastic contacts generate a full triangulated contact patch from the SDF intersection of the two bodies and pass distributed contact normals directly to the solver.

```
~23 000 hydro faces  →  ~205 solver contacts  =  ~99% reduction
```

The contact patch is rendered live as a pressure-coloured iso-surface: **red = high pressure** (patch centre), **blue = low pressure** (patch edge).

---

## Files

| File | Purpose |
|---|---|
| `rj45_hydro.py` | Scene setup, physics kernels, simulation loop |
| `hydro_contact_viz.py` | Reusable `HydroContactViz` class — ImGui panel and contact patch rendering |

`HydroContactViz` can be dropped into any Newton example that uses a `CollisionPipeline` with `output_contact_surface=True`.

---

## Run

```powershell
python examples/rj45_hydro/rj45_hydro.py
```

| Control | Action |
|---|---|
| Gizmo arrows | Drag plug into socket |
| Right-drag | Spring-pick any body |
| `H` | Toggle ImGui panel |
| `Space` | Pause / resume |

The ImGui panel shows live metrics and sliders for **Opacity**, **Density**, and **Gamma**.

---

## Tests

```powershell
python -m pytest examples/rj45_hydro/tests/ -v
```

37 tests — no GPU or Newton install required.
