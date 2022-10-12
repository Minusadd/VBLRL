# VBLRL


Source code for Model-based Lifelong Reinforcement Learning with Bayesian Exploration (NeurIPS 2022).
>We propose a model-based lifelong reinforcement-learning approach that estimates a hierarchical Bayesian posterior distilling the common structure shared across different tasks. The learned posterior combined with a sample-based Bayesian exploration procedure increases the sample efficiency of learning across a family of related tasks. We first derive an analysis of the relationship between the sample complexity and the initialization quality of the posterior in the finite MDP setting. We next scale the approach to continuous-state domains by introducing a Variational Bayesian Lifelong Reinforcement Learning algorithm that can be combined with recent model-based deep RL methods, and that exhibits backward transfer. Experimental results on several challenging domains show that our algorithms achieve both better forward and backward transfer performance than state-of-the-art lifelong RL methods.


The main dependencies are basically the same as [LPG-FTW](https://github.com/Lifelong-ML/LPG-FTW), except that:

- We use [MetaWorld v2](https://github.com/rlworkgroup/metaworld) for MT10 and MT50 domains, which requires [Mujoco](https://www.roboti.us/index.html) 2.1. For the other domains, we use Mujoco 1.5.
- Install the modified `gym` and `gym-extensions-mod` using:
```
$ cd gym
$ pip install -e .
$ cd ../gym-extensions-mod
$ pip install -e .
```
To reproduce a experiment, run: `python launch_experiment_[ENV_NAME].py ./configs/[ENV_NAME].json`. By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.
Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date. The file `progress.csv` contains statistics logged over the course of training.