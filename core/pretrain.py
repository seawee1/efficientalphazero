import os

from statistics import mean

import time

import ray
import torch

from config.base import BaseConfig
from core.replay_buffer import ReplayBuffer, TransitionBuffer
from core.workers import DemonstrationWorker


def create_filled_demonstration_buffer(args, config):
    demonstration_buffer = ReplayBuffer.remote(config.demo_buffer_size)
    demo_workers = [
        DemonstrationWorker.options(
            num_cpus=args.num_cpus_per_worker,
            num_gpus=args.num_gpus_per_worker).remote(config, demonstration_buffer)
        for _ in range(args.num_rollout_workers)
    ]
    demo_workers = [demo_worker.run.remote() for demo_worker in demo_workers]

    while True:
        num_demo = ray.get(demonstration_buffer.size.remote())
        print(f"Collected {num_demo} demonstration transitions...")
        if num_demo >= config.demo_buffer_size:
            print("Collection done")
            break
        time.sleep(5)
    ray.wait(demo_workers)
    return demonstration_buffer


def pretrain(args, config: BaseConfig, model, summary_writer, log_dir):
    print("Starting pre-training...")
    ray.init()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.1, total_iters=config.pretrain_steps * config.num_sgd_iter, verbose=True)

    model.train()

    demonstration_buffer = create_filled_demonstration_buffer(args, config)

    print(f"Pre-training for {config.pretrain_steps} steps...")
    for train_step in range(config.pretrain_steps):
        print(f"Training step {train_step}...")
        if train_step >= config.pretrain_steps:  # Check if we are done
            break

        # Do optimization step
        total_losses, policy_losses, value_losses = [], [], []
        for i in range(config.num_sgd_iter):
            print(f"SGD step {i}...")
            train_batch, _ = ray.get(demonstration_buffer.sample.remote(config.batch_size, config.frame_stack))
            total_loss, policy_loss, value_loss = model.update_weights(train_batch, optimizer, scaler, scheduler)
            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # Broadcast weights
        summary_writer.add_scalar('pretrain/total_loss', mean(total_losses), train_step)
        summary_writer.add_scalar('pretrain/policy_loss', mean(policy_losses), train_step)
        summary_writer.add_scalar('pretrain/value_loss', mean(value_losses), train_step)

    print("Pre-training finished!")
    torch.save(model.state_dict(), os.path.join(log_dir, f'model_pretrained.pt'))
    ray.shutdown()

