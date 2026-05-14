#!/usr/bin/env bash
# Rigorous cold+warm compile benchmark.
# - Clears BOTH Warp source-hash cache AND CUDA NVRTC ComputeCache
#   before every cold run for a true cold start.
# - Randomizes solver order within each iteration to spread OS file-cache
#   warmth across all solvers.
# - Runs N iterations; output is later aggregated for median + IQR.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

WARP_CACHE="${WARP_CACHE:-$LOCALAPPDATA/NVIDIA/warp/Cache/1.13.0}"
NVRTC_CACHE="${NVRTC_CACHE:-$APPDATA/NVIDIA/ComputeCache}"
LOG="$RESULTS_DIR/solver_rigorous.txt"
CSV="$RESULTS_DIR/solver_rigorous.csv"
SCRIPT="$SCRIPT_DIR/time_example_phases.py"
export NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-C:\\nc}"
N_ITERS="${N_ITERS:-3}"
SOLVERS=(kamino_robot_dr_legs robot_anymal_d cloth_hanging)

: > "$LOG"
echo "iter,solver,kind,import_s,viewer_s,init_s,step_s,render_s,total_s" > "$CSV"

# Fisher–Yates shuffle on a bash array, in place.
shuffle() {
    local i j tmp n=${#SOLVERS[@]}
    for ((i = n - 1; i > 0; i--)); do
        j=$((RANDOM % (i + 1)))
        tmp=${SOLVERS[i]}; SOLVERS[i]=${SOLVERS[j]}; SOLVERS[j]=$tmp
    done
}

# Run one timing pass and parse the phase output into CSV.
run_one() {
    local iter="$1" ex="$2" kind="$3"
    local tmpfile
    tmpfile=$(mktemp)
    echo "===== iter=$iter / $kind / $ex =====" | tee -a "$LOG"
    python "$SCRIPT" "$ex" 2>&1 | tee -a "$LOG" | tee "$tmpfile" >/dev/null
    # Extract totals
    local imp viw ini stp ren tot
    imp=$(awk '/after_import/{print $4; exit}' "$tmpfile")
    viw=$(awk '/after_viewer_ctor/{print $4; exit}' "$tmpfile")
    ini=$(awk '/after_example_init/{print $4; exit}' "$tmpfile")
    stp=$(awk '/after_step/{print $4; exit}' "$tmpfile")
    ren=$(awk '/after_render/{print $4; exit}' "$tmpfile")
    tot=$(awk '/^TOTAL:/{print $2; exit}' "$tmpfile" | tr -d 's')
    echo "$iter,$ex,$kind,${imp:-NA},${viw:-NA},${ini:-NA},${stp:-NA},${ren:-NA},${tot:-NA}" >> "$CSV"
    rm -f "$tmpfile"
}

# Wipe caches that influence "cold" compile.
clear_cold_caches() {
    rm -rf "$WARP_CACHE"
    rm -rf "$NVRTC_CACHE"
}

for iter in $(seq 1 "$N_ITERS"); do
    shuffle
    echo "########## ITER $iter / order: ${SOLVERS[*]} ##########" | tee -a "$LOG"
    for ex in "${SOLVERS[@]}"; do
        clear_cold_caches
        run_one "$iter" "$ex" cold
        run_one "$iter" "$ex" warm
    done
done

echo "Done. Raw log: $LOG"
echo "CSV: $CSV"
