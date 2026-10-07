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
DDQN/PER training and digital-twin datasets remain to be implemented.
