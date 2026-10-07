from agents.ddqn import encode_observation
from agents.replay import Transition


def run_episode(env, agent, *, training=False):
    if tuple(env.action_space.powers) != agent.power_levels:
        raise ValueError('Environment and checkpoint power grids differ')
    observations, info = env.reset()
    masks = info['action_masks']
    pending = {}
    total_reward = 0.0
    rates, losses = [], []
    violations = 0
    shortfall = 0.0
    rate_violations = 0
    updates_before = agent.updates
    for _ in range(env.max_ttis):
        actions = {drone: agent.act(observations[drone], mask, training=training)
                   for drone, mask in masks.items()}
        if training:
            for drone, action in actions.items():
                pending[drone] = [encode_observation(observations[drone], agent.power_levels[-1]), action, 0.0, 0]
        next_obs, reward, terminated, truncated, result = env.step_discrete(actions)
        done = terminated or truncated
        next_masks = result['next_action_masks']
        total_reward += reward
        rates.append(result['sum_rate'])
        violations += sum(not checks['interference_constraint'] for checks in result['constraints'].values())
        rate_violations += sum(not checks['rate_constraint'] for checks in result['constraints'].values())
        shortfall += sum(result['rate_shortfalls'].values())
        if training:
            # A waiting drone has no power action. Accumulate discounted global
            # rewards until its next scheduled decision, rather than inventing
            # an all-false next mask for an ordinary one-step transition.
            for drone in list(pending):
                state, action, accumulated, duration = pending[drone]
                accumulated += agent.config.gamma**duration * reward / agent.config.reward_scale
                duration += 1
                if done or drone in next_masks:
                    agent.replay.add(Transition(state, action, accumulated,
                        encode_observation(next_obs[drone], agent.power_levels[-1]),
                        next_masks.get(drone, (False,) * len(agent.power_levels)),
                        0.0 if done else agent.config.gamma**duration))
                    del pending[drone]
                else:
                    pending[drone] = [state, action, accumulated, duration]
            agent.training_ttis += 1
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        observations, masks = next_obs, next_masks
        if done:
            break
    return {'reward': total_reward, 'mean_sum_rate': sum(rates) / len(rates),
            'loss': sum(losses) / len(losses) if losses else None,
            'c3_violations': violations, 'rate_violations': rate_violations,
            'mean_total_shortfall': shortfall / len(rates),
            'penalty_multiplier': env.reward_model.multiplier,
            'optimizer_updates': agent.updates - updates_before,
            'replay_size': len(agent.replay), 'epsilon': agent.epsilon,
            'beta': agent.beta, 'ttis': len(rates)}
