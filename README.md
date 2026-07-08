# MayoRL

MayoRL is a research workspace for reinforcement learning on the Micropolis
city simulator. The current target is a mayor agent that must manage a city
under real budget and tax constraints instead of unlimited construction funds.

## Usage

Micropolis depends on native GTK/engine components, so the supported runtime is
Docker.

Build the image:

```bash
docker build -f Dockerfile.mayorl -t mayorl:latest .
```

For diagnostics, do not mount the whole repository over `/usr/src/app`; that can
hide compiled engine artifacts from the image. From WSL2, mount only the files
being edited:

```bash
docker run -d --name mayorl-diag \
  -v "$(pwd)/mayorl:/usr/src/app/mayorl" \
  -v "$(pwd)/gym_city/envs/corecontrol.py:/usr/src/app/gym_city/envs/corecontrol.py" \
  -v "$(pwd)/gym_city/envs/tilemap.py:/usr/src/app/gym_city/envs/tilemap.py" \
  mayorl:latest tail -f /dev/null
```

Verify tax collection with a clean powered city:

```bash
docker exec mayorl-diag sh -c 'cd /usr/src/app && xvfb-run -a python3 -u -m mayorl.verify_tax'
```

Run a short PPO training job:

```bash
docker exec mayorl-diag sh -c 'cd /usr/src/app && xvfb-run -a python3 -u -m mayorl.train --algo ppo --map-width 16 --num-processes 1'
```

## What Changed

The original gym-city training loop reset city funds every step, which made
construction effectively unlimited. MayoRL changes the environment into a
budget-constrained problem:

- city funds are no longer reset every step;
- construction costs are paid from actual simulator funds;
- tax income is collected through the Micropolis simulation callbacks;
- unaffordable actions are treated as invalid/no-op actions;
- bankruptcy can terminate an episode;
- observations include budget-related state;
- rewards include population growth, fiscal health, mayor rating, and
  infrastructure coverage;
- `mayorl.verify_tax` uses a clean powered-city builder instead of the broken
  `layGrid` helper that produced rubble-heavy zero-population test cities.

Simulation callbacks needed for the fiscal loop are enabled in the working
copy: `cityEvaluation`, `updateDate`, and `changeCensus`. Disaster/time-event
callbacks remain disabled for now so the first learning checks focus on budget
and tax behavior.

## Current Plan

The immediate goal is to verify short learning runs under the tax constraint:
reward curves, bankruptcy rate, and population growth should move in a
reasonable direction before any VLM mayor/market loop is added.

Known next cleanup items:

- remove inherited action-space hard-coding so the policy uses the environment
  action count directly;
- replace post-hoc invalid-action masking with a learning-aware masking or
  penalty path;
- make observation normalization consistent across map, RCI, and budget
  channels;
- check whether annual tax collection is too sparse for the current episode
  length;
- move the remaining dependencies that `mayorl/` imports from the repository
  root into `mayorl/` so the package becomes self-contained;
- keep `mayorl/vlm_mayor.py` and related VLM paths preserved but inactive until
  the budget-learning baseline is validated.

## Repository Notes

Important paths:

- `mayorl/`: budget-constrained environment, reward, curriculum, training, and
  diagnostics.
- `gym_city/`: inherited Micropolis/gym-city code still used by MayoRL.
- `Dockerfile.mayorl`: Docker runtime for MayoRL.
- `AGENTS.md`: current working notes and project direction.

The Game of Life code inherited from the older repository is not part of the
current MayoRL workflow.

## License And Attribution

This repository contains code under multiple open-source licenses.

- The inherited RL code at the repository root is distributed under the MIT
  License. See `LICENSE`.
- The Micropolis simulator code under `gym_city/envs/micropolis/` is distributed
  under the GNU GPL, version 3 or later, with additional terms from the original
  Micropolis release. Keep those copyright and license notices with any shared
  copy or modified version.
