import os

import time
from statistics import mean, median

import torch
import ray

from config.base import BaseConfig
from core.pretrain import create_filled_demonstration_buffer
from core.workers import RolloutWorker, TestWorker, DemonstrationWorker
from core.replay_buffer import ReplayBuffer, TransitionBuffer
from core.storage import SharedStorage


def train(args, config: BaseConfig, model, summary_writer, log_dir):
    print("Starting training...")
    ray.init()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.1,
                                                  total_iters=config.training_steps * config.num_sgd_iter)

    demonstration_buffer = None
    # if config.demo_buffer_size > 0:  # Uncomment for AlphaTensor like training
    #    demonstration_buffer = create_filled_demonstration_buffer(args, config)

    model.train()

    replay_buffer = ReplayBuffer.remote(config.replay_buffer_size)
    storage = SharedStorage.remote(config, args.amp)
    storage.set_weights.remote(model.get_weights())  # Broadcast model

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=args.num_cpus_per_worker,
            num_gpus=args.num_gpus_per_worker).remote(config, args.device_workers, args.amp, replay_buffer, storage)
        for _ in range(args.num_rollout_workers)
    ]

    workers = [rollout_worker.run.remote() for rollout_worker in rollout_workers]

    storage.set_start_signal.remote()

    for train_step in range(config.training_steps):
        print(f"Training step {train_step}...")
        if train_step >= config.training_steps:  # Check if we are done
            time.sleep(30)
            break

        while True:  # Wait until RolloutWorkers collected their samples
            workers_finished = ray.get(storage.get_workers_finished.remote())
            if workers_finished != args.num_rollout_workers:
                print(f'{workers_finished}/{args.num_rollout_workers} workers finished...')
                time.sleep(10)
                continue
            break

        replay_buffer_size = ray.get(replay_buffer.size.remote())
        print(f"{replay_buffer_size} num samples inside replay buffer...")

        # Do optimization step
        total_losses, policy_losses, value_losses = [], [], []
        print("Updating weights...")
        for i in range(config.num_sgd_iter):
            print(f"SGD step {i}...")
            # if demonstration_buffer is None:  # Uncomment for AlphaTensor-like training
            train_batch, _ = ray.get(replay_buffer.sample.remote(config.batch_size, config.frame_stack))
            # else:
            #    train_batch, _ = ray.get(replay_buffer.sample.remote(int(config.batch_size * 0.7), config.frame_stack))
            #    demo_batch, _ = ray.get(demonstration_buffer.sample.remote(int(config.batch_size * 0.3), config.frame_stack))
            #    train_batch.fuse_inplace(demo_batch)

            total_loss, policy_loss, value_loss = model.update_weights(train_batch, optimizer, scaler, scheduler)
            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # Broadcast weights
        if train_step % config.model_broadcast_interval == 0:
            print("Broadcasting current model...")
            storage.set_weights.remote(model.get_weights())
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_{train_step}.pt'))
            if config.clear_buffer_after_broadcast:
                replay_buffer.clear.remote()

        rollout_worker_logs = ray.get(storage.pop_rollout_worker_logs.remote())

        summary_writer.add_scalar('train/total_loss', mean(total_losses), train_step)
        summary_writer.add_scalar('train/policy_loss', mean(policy_losses), train_step)
        summary_writer.add_scalar('train/value_loss', mean(value_losses), train_step)
        summary_writer.add_scalar('train/replay_buffer_size', replay_buffer_size, train_step)

        TransitionBuffer.log(summary_writer, train_step, rollout_worker_logs, prefix='rollout')

        storage.reset_workers_finished.remote()
        storage.incr_counter.remote()

    ray.wait(workers)
    print("Training finished!")
    torch.save(model.state_dict(), os.path.join(log_dir, f'model_latest.pt'))
    ray.shutdown()
