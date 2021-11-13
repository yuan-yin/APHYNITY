# Official code of APHYNITY

[*Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting*](https://arxiv.org/abs/2010.04456) (ICLR 2021, Oral)

Yuan Yin\*, Vincent Le Guen\*, Jérémie Dona\*, Emmanuel de Bézenac\*, Ibrahim Ayed*, Nicolas Thome, Patrick Gallinari 

(*Equal contribution)

> Forecasting complex dynamical phenomena in settings where only partial knowledge of their dynamics is available is a prevalent problem across various scientific fields. While purely data-driven approaches are arguably insufficient in this context, standard physical modeling based approaches tend to be over-simplistic, inducing non-negligible errors. In this work, we introduce the APHYNITY framework, a principled approach for augmenting incomplete physical dynamics described by differential equations with deep data-driven models. It consists in decomposing the dynamics into two components: a physical component accounting for the dynamics for which we have some prior knowledge, and a data-driven component accounting for errors of the physical model. The learning problem is carefully formulated such that the physical model explains as much of the data as possible, while the data-driven component only describes information that cannot be captured by the physical model, no more, no less. This not only provides the existence and uniqueness for this decomposition, but also ensures interpretability and benefits generalization. Experiments made on three important use cases, each representative of a different family of phenomena, i.e. reaction-diffusion equations, wave equations and the non-linear damped pendulum, show that APHYNITY can efficiently leverage approximate physical models to accurately forecast the evolution of the system and correctly identify relevant physical parameters.

## Usage

```
python3 train_aphynity.py [-h] [-r ROOT] [-p PHY] [--aug | --no-aug] [-d DEVICE] dataset
```

You can choose `rd`, `wave`, or `pendulum` dataset.

### Options
- Choose physical model:
  - `--phy incomplete` (default): incomplete Param PDE
  - `--phy complete`: complete Param PDE
  - `--phy true`: true Param PDE
  - `--phy none`: no physics
- Augmentation:
  - `--aug` (default): enable augmentation
  - `--no-aug`: disable augmentation
- Choose device:
  - `--device cpu` (default): run on CPU
  - `--device cuda:X`: run on CUDA compatible GPU
- Choose root path for experiments with `--root ROOT`

### Example
Run APHYNITY with `rd` dataset:

```
python3 train_aphynity.py rd --phy incomplete --aug
```