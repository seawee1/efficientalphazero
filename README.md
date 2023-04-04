# (Efficient) AlphaZero
An efficient and clean implementation of AlphaZero for single-player domains in PyTorch. The implementation is inspired by the awesome [EfficientZero](https://github.com/YeWR/EfficientZero) implementation, a derivative work building [muZero](https://arxiv.org/abs/1911.08265). Another invaluable resource was the [alphazero_singleplayer](https://github.com/tmoer/alphazero_singleplayer) repository and the corresponding blogpost.


## Features
- Worker parallelization using [Ray](https://www.ray.io/)
- Model inference parallelism via Batch MCTS
- [AMP](https://developer.nvidia.com/automatic-mixed-precision) support
- A lot of improvements used in muZero like min-max value scaling and discrete value support for intermediate rewards during MCTS
- Model pre-training and training data enrichment through demonstrations (similar to [AlphaTensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor))
- Easily extendable to new singleplayer environments (just sub-class the BaseConfig)

## Setup
Run
```pip install -r requirements.txt```
and
```conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia```

## Example usage
To train AlphaZero on CartPole, run:

```
python main.py --env cartpole --opr train,test
```
