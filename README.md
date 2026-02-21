# Bachelor_project_MST

Code and analysis pipeline for the bachelor thesis **“From dyads to groups: extending coupled-oscillator models of turn-taking to structured multi-agent scenarios”** (Aarhus University).

## What this repo does
I use a Kuramoto-style coupled phase-oscillator model as a minimal baseline for **timing coordination**. Anti-phase locking (phase difference near π) is treated as a proxy for turn-taking-like alternation (not a model of discrete turns).

Scenarios:
- **Dyad (N=2)**: sweep coupling strength `k` × frequency mismatch `Δω`
- **Triad (N=3)**: presets (`all_to_all`, `coalition_01`, `leader_0`) sweeping `Δω_tri` × `k_weak` (with fixed `k_strong`)
- **Classroom (N=1+M)**: teacher–student asymmetry sweeping `k_ts`, `k_st`, `k_ss`

## Core model

For oscillator $i$:

$$
\dot{\theta}_i(t)=\omega_i+\sum_{j=1}^{N}K_{ij}\sin\!\big(\theta_j(t)-\theta_i(t)\big).
$$

Convention: $K_{ij}$ is influence from oscillator $j$ to oscillator $i$
(row $i$ receives from column $j$).

## Outputs
Sweep runs write timestamped folders with:
- configuration (`config_used.json`)
- per-run and per-condition tables (`*_sweep_runs.csv`, `*_sweep_conditions.csv`)
- figures (heatmaps + exemplar runs)

## Reproducibility settings
- RK4, `dt=0.01`, `t_max=120s`
- seeds `{1,2,3,4,5}`
- tail means over the final 30%
- PLV window = 2.0s (`W=200`)

## How to run (high level)
1. Run Python scripts to generate sweep outputs in `final_outputs/`
2. Run R scripts in `analysis/` to make heatmaps and summary tables from `*_sweep_conditions.csv`
   
## Conact
Manuela Skov Thomasen: 202107872@post.au.dk
