# Drone communication simulation

## Structure

- `environment/config.py`: validated network and communication settings.
- `environment/drone_env.py`: scheduling, transmission and episode orchestration.
- `environment/actions.py`: discrete power grid, C3 masks and feasible action selection.
- `environment/reward.py`: cooperative reward settings and adaptive rate penalty.
- `environment/channel_provider.py`: replaceable all-link channel snapshots.
- `environment/channel.py`: signal, interference and SINR calculations.
- `environment/mobility.py`: waypoint missions and TTI/slot timing.
- `environment/scheduler.py`: proportional-fair scheduling.
- `environment/rate.py`: finite-blocklength rates.
- `examples/`: runnable baseline episodes.
- `tests/`: behavioral and integration tests.

## Run from the project root

```powershell
python -B -m unittest discover -s tests -v
python -B -m examples.fixed_power_episode
python -B -m examples.masked_power_episode
```

## Power actions and masking

`step({drone_id: watts})` retains the continuous-power baseline interface.
It checks power bounds but permits C3 violations for baseline comparisons.
`step_discrete({drone_id: index})` maps indices to configured powers and rejects
actions that exceed C3. Both require exactly the scheduled drone IDs.

`NetworkConfig.num_power_levels` defaults to 20. Levels are linearly spaced
between minimum and maximum power; spacing is a reproduction assumption.
Masks test candidate power times the largest cross-cell channel gain against
the interference threshold. Serving-link gain is excluded. Equality is feasible.

`reset()` info includes `power_levels` and `action_masks` for the first decision.
Step info includes current masks and `next_action_masks` aligned with
`next_rrb_assignment` and returned observations. Copy these next masks into
future replay transitions so DDQN target action selection can also be masked.

`PowerActionSpace.select()` supports greedy selection and epsilon-greedy
exploration using an explicit random generator. Both choose only feasible
actions. This helper is not a trained policy. An empty mask raises
`NoFeasibleActionError`; no unsafe fallback or implicit outage action is added.
Idle drones have no action mask because they do not need a power decision.

## Cooperative reward

Both step APIs return one global reward: `sum_rate - multiplier * total_shortfall`.
Shortfall is `max(0, min_rate - achieved_rate)`, in bits/s. Reward uses the current
multiplier; after a completed TTI it increases by `learning_rate * mean_shortfall`.
Reading metrics does not change it, failed transmissions do not update it, and
reset restores the configured initial multiplier. Manual `advance_tti()` also
updates it because that method completes a transmission.

`RewardConfig` defaults to cooperative mode, initial multiplier 0, learning rate
1e-6, and penalties for all drones including unscheduled drones. Initialization
and update rate are implementation assumptions, not recovered paper parameters.
The update rate has units inverse to bits/s; no hidden Mbps conversion is applied.
The first reward has no penalty with multiplier zero. Persistent violations raise
the multiplier; it stays constant when all evaluated rates are satisfied.

The paper is ambiguous about penalizing unscheduled drones. Set
`penalize_unscheduled=False` to evaluate only scheduled drones, with their count
as the denominator for mean shortfall. Info still reports raw `rate_shortfalls`
for all drones, while `penalty_shortfalls` shows exactly which enter the reward.
An empty evaluated set has zero mean shortfall.

```python
from environment.reward import RewardConfig
from environment.drone_env import DroneCommunicationEnvironment

env = DroneCommunicationEnvironment(reward_config=RewardConfig(
    initial_multiplier=1.0, learning_rate=1e-6,
    penalize_unscheduled=True,
))
```

Use `RewardConfig(mode='sum_rate')` for an unpenalized baseline, or
`adaptive=False` to freeze the configured multiplier for evaluation. Info reports
sum rate, penalty, reward, current/next multipliers and mean/total shortfall.
Penalties encourage rate satisfaction but cannot make infeasible scenarios feasible.
## Shared DDQN (Step 8)

- `agents/ddqn.py`: shared evaluation/target networks, fixed feature transforms,
  masked epsilon-greedy inference, optimizer updates and policy checkpoints.
- `agents/replay.py`: uniform and prioritized replay, and immutable transition records.
- `training/runner.py`: shared experiences from all drones and evaluation episodes.
- `training/train_ddqn.py`: runnable training command and JSON metrics.

```powershell
python -m pip install -r requirements-rl.txt
python -B -m training.train_ddqn --episodes 5 --ttis 30
```

Outputs go to ignored `artifacts/ddqn/`: `policy.pt` and `metrics.json`.
The checkpoint supports loaded greedy inference and saves optimizer/counters;
replay and RNG state are not saved, so it is not an exact training-resume snapshot.
Network/reward configuration metadata is saved by the training command.

The shared CPU network has four local inputs and one output per power level.
Two 64-unit ReLU layers, exploration decay, transforms, gradient clipping and
reward scaling are implementation assumptions. Default learning rate 5e-4,
discount 0.9, memory 50000 and batch 32 follow the paper's table. Target copying
every 100 optimizer updates is the documented interpretation of its interval.
Inputs use log-scaled channel/SINR, rate in Mbps and power divided by maximum;
global rewards are divided by 1e6 consistently before entering replay.

DDQN chooses the feasible next action with the evaluation network and evaluates
it with the target network. No masked action contributes to a bootstrap target.
Waiting drones have no power decision: their transitions accumulate discounted
global rewards until the next scheduled decision, with gamma raised to the
number of elapsed TTIs. This is a documented extension for intermittent scheduling.
The runner treats the configured episode horizon as a finite return boundary
(zero bootstrap), even though the environment labels the limit a truncation.
For a continuing-task experiment this boundary handling must be changed.
Empty feasible masks raise an explicit error rather than inventing an outage action.

Evaluation uses greedy actions and does not modify the agent or replay. The
environment's penalty adapts unless RewardConfig(adaptive=False) is selected.
The default training scene is stationary and simplified; a short successful run
validates plumbing, not convergence or paper-level performance.

## Prioritized experience replay (Step 9)

PER is enabled by default. Priority is `abs(TD error) + epsilon`, sampling
probability is proportional to `priority**alpha`, and the mean squared loss is
weighted by `(N * probability)**(-beta)`. Weights are normalized by the maximum
over the whole buffer. Sampling uses replacement, so a batch may repeat entries;
duplicate priority updates keep the largest absolute TD error.

Defaults follow the paper: alpha 0.6, epsilon 1e-6, initial beta 0.4. Beta
increases linearly to 1 across 10000 optimizer updates; that duration is an
implementation assumption. New transitions receive the largest current priority,
a standard PER insertion choice rather than a fresh TD calculation. Priorities
are updated using the pre-optimizer TD errors of sampled transitions. Masked
DDQN targets and waiting-drone transitions remain unchanged.

For a uniform-replay baseline:

```powershell
python -B -m training.train_ddqn --uniform-replay --output artifacts/uniform
```

The reference sampler computes probabilities in O(buffer size). A sum-tree
implementation remains an optimization if long-run profiling shows a bottleneck.
Checkpoints save PER settings and update count (hence beta), but not replay
contents/priorities or RNG state. Digital-twin construction is the next major step.

## Complete training loop

```powershell
python -B -m training.train_ddqn --episodes 100 --ttis 100 --eval-interval 10 --eval-episodes 3 --output artifacts/run_01
```

Each episode resets the simulator, applies joint masked power decisions, pools
drone experiences, performs a replay update per TTI once a full batch is available,
and logs rate, reward, loss, epsilon, beta, shortfalls and constraint violations.
Exploration decays over 10000 training TTIs by CLI default. Replay warmup means
collecting at least batch-size entries before the first optimizer update.

`training/loop.py` periodically evaluates greedy actions in a separate environment
with a fixed penalty multiplier. Evaluation does not update weights, replay or
exploration counters. The default scene is deterministic, so repeated evaluations
do not represent independent channel samples. For varied scenarios, supply distinct
training/evaluation environments to `train()`.

Outputs: `training.csv`, incrementally saved `metrics.json`, `run_config.json`,
`latest.pt`, `best.pt` and final `policy.pt`. Best selection uses greedy mean sum
rate; minimum-rate satisfaction is reported separately and is not guaranteed.
`best.pt` is reloaded for final evaluation. Checkpoints are for inference or warm
starts, not exact continuation; replay/RNG state is not persisted. Existing runs
are protected: choose a fresh output directory. No automatic early stopping is
applied because noisy sum-rate changes do not establish convergence.
