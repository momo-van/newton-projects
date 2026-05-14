# VBD vs MuJoCo — Top Learnings from Tuning a Tight-Tolerance Insertion Task

DIN 5480 z=6 spline-hub assembly · Newton physics engine · RTX 5000 Ada

---

## Recommendation

**Use VBD for snap-fit and press-fit assembly tasks. Use MuJoCo for everything else.**

The spline insertion task requires a discrete positional transition: teeth must physically drop
into grooves when alignment is found. VBD's position-level contacts deliver that — penetration
is corrected directly, so the shaft falls in at alignment. MuJoCo's force-level contacts always
oppose the descent, producing a resisting force that grows with penetration. The shaft is
continuously pushed back out; alignment produces no preferential drop. This is not a MuJoCo bug
— it is the correct behaviour of an impulse-based solver. It is simply the wrong model for this
task geometry.

**Choose MuJoCo when:** rigid impact, free-object manipulation, locomotion, or any task where
force-level contacts are physically correct and the 1.40× per-substep speed advantage matters
for batch rollouts.

**The one real cost of VBD here:** torsion drift of 20.95 °/s during post-insertion lock.
Position-level groove walls have finite stiffness. For hard torsion-lock requirements, increase
`kh` or enforce a constraint-based lock after seating.

---

## 1. Contact model determines whether groove snap is physically possible

VBD resolves contacts at the **position level** — penetration is directly corrected each substep.
MuJoCo resolves at the **force level** — a force opposes penetration.

For a snap-to-engage insertion, the shaft needs to physically drop into the groove once tooth-gap
alignment is found. Position-level contacts allow that naturally. Force-level contacts resist axial
descent by design. No stiffness tuning changes this — it is a fundamental property of the model.

---

## 2. Contact stiffness values are not comparable across solvers

- VBD `kh = 5×10⁶` — displacement penalty (units: N/m applied to position correction)
- MuJoCo `kh = 2×10⁹` — force-level stiffness (units: N/m in an impulse formulation)

The 400× difference is not a tuning choice. They are different physical quantities operating in
different constraint formulations. Treating them as equivalent produces either jitter (too soft)
or a rigid wall (too stiff), and the mapping between them is geometry- and timestep-dependent.

---

## 3. Search force and contact stiffness must be co-designed

At MuJoCo `kh = 2×10⁹`, a **15 N descent force** reaches contact equilibrium at ~2 mm bore
penetration. The groove detection threshold requires **5 mm** of entry. MuJoCo is not failing —
it is correctly balancing forces. The shaft is held out by the contact before it can reach the
groove. To fix this you must either lower `kh`, increase search force, or accept that force-level
contacts will always resist axial descent before groove entry for this geometry and tolerance.

---

## 4. SDF voxel resolution gates contact quality in VBD — there is a hard rule of thumb

VBD uses a voxelized SDF to compute penetration depth and contact normal. If the voxel pitch is
larger than the feature you are trying to resolve, two things break:

- **Contact normals on tooth flanks get quantized** to the nearest voxel face, producing wrong
  friction direction during the angular search phase.
- **The groove-entry signal is blurred** — the shaft can slip through groove centerline without
  the sharp resistance-drop that triggers snap detection.

**Rule of thumb:** SDF voxel pitch ≤ 1/4 of the smallest contact feature you need to resolve.
For DIN 5480 z=6 m=4 mm (tooth flank detail at ~1–2 mm scale), use ≤ 0.5 mm voxels.
At 44³ voxels over a 64 mm hub diameter the pitch is ~1.45 mm — borderline, and sufficient only
because groove width (~4 mm) is much larger than the voxels. Going coarser loses the snap signal.
MuJoCo uses mesh-mesh GJK/MPR collision and does not have this voxelization constraint.

---

## 5. `njmax` undersizing causes silent wrong dynamics

MuJoCo with elliptic cone friction allocates **3 constraint rows per contact pair**
(1 normal + 2 tangential). With 132 active contacts, the solver needs `njmax ≥ 396`.

Running with `njmax = 132` caused a silent `nefc` overflow — no error was raised, the solver
ran to completion, and produced physically incorrect results. The diagnostic is unexpected
near-zero contact forces despite visible penetration.

**Rule of thumb:** `njmax = 3 × max_contacts + margin`. For this task, 512 is safe.

---

## 6. Equal substeps, not equal iterations, is the correct comparison basis

- VBD: 8 substeps × 32 iterations = **256 work units / frame**, mean frame time 71.6 ms, per-substep 8.94 ms
- MuJoCo: 8 substeps × 15 iterations = **120 work units / frame**, mean frame time 51.3 ms, per-substep 6.41 ms

MuJoCo's **1.40× frame speedup** partly reflects doing less work per frame. Per-substep time
is the apple-to-apple metric. Any benchmark that compares only wall time without normalising
by work units conflates solver efficiency with solver configuration.

---

## Final Tuning — VBD

- **`kh = 5×10⁶`** — position-level displacement penalty. Start here; increase if teeth pass through each other.
- **Substeps = 8** — stable at 60 fps with `sub_dt = 2.08 ms`. Increase for higher-speed impacts.
- **Iterations = 32** — more iterations improve convergence in dense contact; diminishing returns above 48.
- **SDF resolution ≥ 64³ narrow-band** — voxel pitch ≤ 1/4 of smallest contact feature. 44³ is borderline for 4 mm module teeth.
- **Collision once / frame** — sufficient for slow assembly; may miss fast impact events.
- **`descent_k = 800 N/m`** — spring-damper axial guidance. Too high → oscillation; too low → shaft drifts laterally.
- **`lateral_k = 1×10⁵ N/m`** — keeps shaft aligned with bore axis during search.
- **`μ = 0.25`** — dry steel-on-steel. Higher values stall angular search; lower values cause torsion slip post-insertion.
- **Warm-up = 5 frames** — exclude from all timing stats; Warp JIT compilation inflates first-frame cost.

**Watch-outs:**
- Torsion drift during lock phase (~20 °/s) is a position-level stiffness limit, not a solver failure.
- SDF normal quantization on tooth flanks produces incorrect friction direction if voxel pitch > groove width / 4.
- `body_f` is always applied explicitly — there is no implicit stabilization of applied forces.

---

## Final Tuning — MuJoCo

- **`kh = 2×10⁹`** — force-level stiffness. Not comparable to VBD `kh`. Higher = stiffer contact wall.
- **Substeps = 8** — match VBD for fair comparison. `sub_dt = 2.08 ms`.
- **Iterations = 15** — MuJoCo converges faster per iteration than VBD on this task.
- **`njmax = 512`** — must be ≥ 3 × max active contacts. Elliptic cone = 3 rows/pair. Undersizing is completely silent.
- **Collision once / substep** — MuJoCo detects contacts 8× more often than VBD at equal substeps.
- **`descent_k = 800 N/m`** — same as VBD; identical trajectory script.
- **`μ = 0.25`** — same as VBD.
- **Warm-up = 5 frames** — same as VBD.

**Watch-outs:**
- Force-level contacts balance the 15 N search force at ~2 mm bore penetration. Groove detection at 5 mm is never reached. Fix: increase search force above `kh × detection_depth` or lower `kh`.
- Torsion is perfectly stiff post-insertion (0 °/s drift) — groove walls act as rigid constraints.
- `njmax` overflow is completely silent. Verify with a contact-count diagnostic before trusting dynamics output.
- Per-substep time is 1.40× faster than VBD at equal substeps — the better choice for batch rollouts on tasks that do not require groove snap.
