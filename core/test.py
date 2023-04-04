from statistics import mean, median
import os

from ray.util.client import ray
import yaml

from core.storage import SharedStorage, add_logs
from core.workers import RolloutWorker, TestWorker


def test(args, config, model, log_dir):
    print("Starting testing...")
    ray.init()

    test_workers = [
        TestWorker.options(
            num_cpus=args.num_cpus_per_worker,
            num_gpus=args.num_gpus_per_worker).remote(config, args.device_workers, args.amp)
        for _ in range(args.num_rollout_workers)
    ]
    num_episodes_per_worker = int(args.num_test_episodes / args.num_rollout_workers)
    workers = [test_worker.run.remote(model.get_weights(), num_episodes_per_worker) for test_worker in test_workers]

    ray.wait(workers)

    test_stats_all = {}  # Accumulate test stats
    for i, test_worker in enumerate(test_workers):
        test_stats = ray.get(test_worker.get_stats.remote())
        add_logs(test_stats, test_stats_all)

    accum_stats = {}  # Calculate stats
    for k, v in test_stats_all.items():
        accum_stats[f'{k}_mean'] = float(mean(v))
        accum_stats[f'{k}_median'] = float(median(v))
        accum_stats[f'{k}_min'] = float(min(v))
        accum_stats[f'{k}_max'] = float(max(v))
    print(yaml.dump(accum_stats, allow_unicode=True, default_flow_style=False))

    with open(os.path.join(log_dir, 'result.yml'), 'w') as yaml_file:  # Write to file
        yaml.dump(accum_stats, yaml_file, default_flow_style=False)

    print("Testing finished!")

    ray.shutdown()
