"""Multi-episode training with separate evaluation and durable run outputs."""
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from agents.ddqn import SharedDDQN
from training.runner import run_episode


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 100
    eval_interval: int = 10
    eval_episodes: int = 3
    checkpoint_interval: int = 10

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Training counts and intervals must be positive integers')


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def evaluate(env, agent, episodes):
    results = [run_episode(env, agent, training=False) for _ in range(episodes)]
    return {
        'mean_sum_rate': sum(row['mean_sum_rate'] for row in results) / episodes,
        'mean_reward': sum(row['reward'] for row in results) / episodes,
        'c3_violations': sum(row['c3_violations'] for row in results),
        'rate_violations': sum(row['rate_violations'] for row in results),
        'episodes': episodes,
    }


def train(train_env, eval_env, agent, output, config=None):
    config = config or TrainingConfig()
    if train_env is eval_env:
        raise ValueError('Evaluation must use a separate environment')
    if eval_env.reward_model.config.adaptive:
        raise ValueError('Evaluation penalty must be frozen for comparable scores')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_config.json').exists():
        raise FileExistsError('Output already contains a run; choose a new directory')
    metadata = {
        'training': asdict(config), 'agent': asdict(agent.config),
        'network': asdict(train_env.config), 'reward': asdict(train_env.reward_model.config),
        'evaluation_reward': asdict(eval_env.reward_model.config),
        'max_ttis': train_env.max_ttis, 'ttis_per_slot': train_env.mobility.num_ttis_per_ts,
        'throughput_smoothing': train_env.throughput_smoothing,
        'waypoints': train_env.mobility.waypoints,
        'channel_provider': type(train_env.channel_provider).__name__,
        'best_selection': 'highest greedy evaluation mean_sum_rate',
    }
    write_json(output / 'run_config.json', metadata)
    history, evaluations = [], []
    best = float('-inf')
    for episode in range(1, config.episodes + 1):
        result = run_episode(train_env, agent, training=True)
        result['episode'] = episode
        history.append(result)
        with (output / 'training.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            if episode == 1:
                writer.writeheader()
            writer.writerow(result)
        if episode % config.eval_interval == 0 or episode == config.episodes:
            score = evaluate(eval_env, agent, config.eval_episodes)
            score['episode'] = episode
            evaluations.append(score)
            if score['mean_sum_rate'] > best:
                best = score['mean_sum_rate']
                agent.save(output / 'best.pt', metadata)
        if episode % config.checkpoint_interval == 0 or episode == config.episodes:
            agent.save(output / 'latest.pt', metadata)
        write_json(output / 'metrics.json', {'training': history, 'evaluations': evaluations})
        loss = 'warming up' if result['loss'] is None else f"{result['loss']:.4f}"
        print(f"Episode {episode}/{config.episodes}: {result['mean_sum_rate'] / 1e6:.3f} Mbps; "
              f"epsilon={agent.epsilon:.3f}; loss={loss}; updates={agent.updates}", flush=True)
    agent.save(output / 'policy.pt', metadata)
    restored = SharedDDQN.load(output / 'best.pt')
    final = evaluate(eval_env, restored, config.eval_episodes)
    write_json(output / 'metrics.json', {'training': history, 'evaluations': evaluations,
                                       'best_policy_evaluation': final})
    return {'training': history, 'evaluations': evaluations, 'best_policy_evaluation': final}
