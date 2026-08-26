# verl-vla

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/verl-project/verl-vla)&nbsp;[![Documentation Status](https://readthedocs.org/projects/verl-vla/badge/?version=latest)](https://verl-vla.readthedocs.io/en/latest/)&nbsp;[![CI](https://img.shields.io/github/actions/workflow/status/verl-project/verl-vla/sanity.yml?branch=main&label=CI)](https://github.com/verl-project/verl-vla/actions/workflows/sanity.yml)&nbsp;[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)&nbsp;[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

**A unified cloud-edge post-training framework for vision-language-action
policies, built on top of [verl](https://github.com/verl-project/verl).**

Large VLA models increasingly place inference and fine-tuning on cloud GPU
clusters, while simulators, physical robots, and human operators may run on
different machines. verl-vla connects these distributed resources into one
post-training system, providing a continuous path from human-in-the-loop data
collection through supervised fine-tuning, policy evaluation, and
reinforcement learning.

Instead of rebuilding the execution stack for every model, environment, or
training algorithm, developers compose reusable workflows on top of a shared
distributed runtime and environment loop. Maintained recipes provide
reproducible starting points that can be adapted to new policies, simulators,
and robot platforms.

[Documentation](https://verl-vla.readthedocs.io/en/latest/) ·
[Quick Start](https://verl-vla.readthedocs.io/en/latest/getting-started/) ·
[Framework Overview](https://verl-vla.readthedocs.io/en/latest/framework-overview/) ·
[Changelog](https://github.com/verl-project/verl-vla/blob/main/CHANGELOG.md)

## One system for the complete post-training loop

![verl-vla architecture](https://raw.githubusercontent.com/verl-project/verl-vla/main/docs/_static/images/architecture.png)

A **workflow** defines the end-to-end procedure and moves data and checkpoints
between stages. A **trainer** advances the selected optimization algorithm.
`TrainCluster` coordinates the distributed workers that execute model training,
rollout, environment interaction, evaluation, recording, and checkpointing.
Together, these layers keep orchestration, algorithms, and distributed
execution separate while allowing them to share one post-training foundation.

### Distributed execution with TrainCluster

`TrainCluster` organizes resources by role rather than physical location.
Actor, rollout, and environment workers can run together on one machine or be
placed across multi-node GPU clusters, simulator hosts, and robot-side devices.
Workflows use the same high-level operations for training, rollout, evaluation,
recording, checkpoint management, and weight synchronization regardless of the
deployment topology.

### Web-based data collection and human intervention

The environment loop publishes observations and accepts controls through a
browser, so the operator's keyboard, gamepad, or XR controller does not need to
be attached to the machine running the simulator or robot. The same interaction
path supports teleoperation, demonstration recording, autonomous rollout, and
human intervention.

This browser-based interaction loop is shared across simulators and physical
robots:

<table>
  <tr>
    <th>Isaac Lab Arena</th>
    <th>LIBERO</th>
    <th>Piper</th>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/verl-project/verl-vla/main/docs/_static/images/teleop-arena.webp" alt="Isaac Lab Arena teleoperation demo" width="360" height="164"></td>
    <td><img src="https://raw.githubusercontent.com/verl-project/verl-vla/main/docs/_static/images/teleop-libero.webp" alt="LIBERO teleoperation demo" width="360" height="164"></td>
    <td><img src="https://raw.githubusercontent.com/verl-project/verl-vla/main/docs/_static/images/teleop-piper.webp" alt="Piper teleoperation demo" width="360" height="164"></td>
  </tr>
</table>

Human intervention is particularly important when policies are deployed
remotely. Cloud inference produces action chunks that the robot executes
locally, while inference and network latency make step-by-step action
replacement impractical. Instead, an operator can interrupt autonomous
execution, provide an arbitrary-length recovery or correction segment, and
then return control to the policy. Policy actions and human corrective actions
remain part of one continuous recorded trajectory.

<p align="center">
  <img src="https://raw.githubusercontent.com/verl-project/verl-vla/main/docs/_static/images/cloud-edge-human-intervention.png" alt="Cloud policy inference sends action chunks to a robot, where a human can intervene during inference and network latency" width="90%">
</p>

### Composable workflows and reproducible recipes

Workflows connect data collection, training, evaluation, and checkpoint
lifecycles into complete procedures. Simple workflows reuse the same execution
layer for teleoperation, SFT, or evaluation; multi-stage algorithms such as
RECAP compose evaluation, trajectory collection, return computation, value
training, advantage annotation, and policy updates without introducing a
separate training stack. Each maintained recipe combines a verified
environment, minimal launcher, configuration, and documentation, with
reference results recorded for validated experiments.

## Supported integrations

| Area | Integrations |
| --- | --- |
| Models | ACT, Gaussian Actor, Pi0.5, and GR00T N1.6 |
| Environments and robots | LIBERO, Isaac Lab Arena, and Piper |
| Training | SFT, SAC-style off-policy training, TD3+BC, FPO, DSRL, and RECAP |
| Human input | Keyboard, gamepad, and XR controller |

Model adapters preserve upstream-native implementations and Hugging Face
checkpoint formats. Environment integrations expose a shared lifecycle and
observation-action contract, allowing models, environments, and workflows to
evolve independently.

## Reproducible recipes

Recipes cover the complete post-training lifecycle from data collection to
fine-tuning and reinforcement learning.

### Data collection

| Environment or robot | Recipes |
| --- | --- |
| LIBERO | [Keyboard](https://verl-vla.readthedocs.io/en/latest/data-collection/libero/keyboard.html) · [Gamepad](https://verl-vla.readthedocs.io/en/latest/data-collection/libero/gamepad.html) · [XR controller](https://verl-vla.readthedocs.io/en/latest/data-collection/libero/xr-controller.html) |
| Isaac Lab Arena | [XR controller](https://verl-vla.readthedocs.io/en/latest/data-collection/isaac-lab-arena/xr-controller.html) |
| Piper | [Keyboard teleoperation and demonstration recording](https://github.com/verl-project/verl-vla/tree/main/examples/data_collection/piper) |

### Fine-tuning

| Model | Recipes |
| --- | --- |
| ACT | [Official LIBERO Spatial demonstrations](https://verl-vla.readthedocs.io/en/latest/fine-tuning/act/official-libero-spatial.html) · [Self-collected LIBERO Spatial demonstrations](https://verl-vla.readthedocs.io/en/latest/fine-tuning/act/self-collected-libero-spatial.html) |
| Gaussian Actor | [LIBERO Spatial task 0](https://verl-vla.readthedocs.io/en/latest/fine-tuning/gaussian-actor/libero-spatial-task0.html) |
| Pi0.5 | [LIBERO Spatial](https://verl-vla.readthedocs.io/en/latest/fine-tuning/pi05/libero-spatial.html) |
| GR00T N1.6 | [LIBERO Spatial](https://verl-vla.readthedocs.io/en/latest/fine-tuning/gr00t/libero-spatial.html) |

### Reinforcement learning

| Method | Recipes |
| --- | --- |
| SAC | [ACT on LIBERO Spatial task 0](https://github.com/verl-project/verl-vla/tree/main/examples/rl/sac/act) · [GR00T N1.6 on the Arena GR1 fridge task](https://github.com/verl-project/verl-vla/tree/main/examples/rl/sac/gr00t) · [GR00T N1.6 on Arena LIBERO](https://github.com/verl-project/verl-vla/tree/main/examples/rl/sac/gr00t) |
| TD3+BC | [Gaussian Actor on LIBERO Spatial task 0](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/td3-bc/gaussian-actor/libero-spatial.html) · [Pi0.5 on LIBERO Spatial task 2](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/td3-bc/pi05/libero-spatial.html) |
| FPO | [Pi0.5 on LIBERO Spatial task 2](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/fpo/pi05/libero-spatial.html) |
| DSRL | [Pi0.5 on LIBERO Spatial tasks 9 and 2](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/dsrl/pi05/libero-spatial.html) · [GR00T N1.6 on all 10 Arena LIBERO Spatial tasks](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/dsrl/gr00t/arena-libero-spatial.html) |
| RECAP | [Pi0.5 on LIBERO-10 task 8](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/recap/pi05/libero10-task8.html) · [GR00T N1.6 on the Arena GR1 fridge task](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/recap/gr00t/arena-gr1.html) |

Support for additional models, environments, training algorithms, and input
devices is under active development.

## Quick start

The following minimal example lets you quickly experience verl-vla's
browser-based teleoperation workflow. Clone the repository, complete the
[verified environment setup](https://verl-vla.readthedocs.io/en/latest/getting-started/#set-up-the-environment),
then activate the local environment and start keyboard teleoperation on the
first LIBERO Spatial task:

```bash
source .venv/bin/activate

vvla-teleop \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[keyboard]'
```

Open [http://localhost:18000](http://localhost:18000) to view the live
teleoperation dashboard. If LIBERO is running on another machine, replace
`localhost` with that machine's hostname or IP address.

Follow the keyboard controls shown in the dashboard to operate the robot arm.
Press Enter to reset the environment and Ctrl+C in the terminal to stop.

Continue with the full
[Quick Start](https://verl-vla.readthedocs.io/en/latest/getting-started/) to
record and replay demonstrations, fine-tune and evaluate an ACT policy, collect
optional DAgger intervention data, and improve the policy with a compact,
single-batch TD3+BC example. The guide also provides an OSMesa CPU-rendering
command for machines without a rendering GPU.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Quick Start](https://verl-vla.readthedocs.io/en/latest/getting-started/) | An end-to-end LIBERO workflow from teleoperation to training and evaluation |
| [Framework Overview](https://verl-vla.readthedocs.io/en/latest/framework-overview/) | Architecture, workflows, trainers, `TrainCluster`, integrations, and resource configuration |
| [Data Collection](https://verl-vla.readthedocs.io/en/latest/data-collection/) | Environment installation and device-specific teleoperation, recording, and intervention examples |
| [Fine-Tuning](https://verl-vla.readthedocs.io/en/latest/fine-tuning/) | Reproducible supervised fine-tuning workflows |
| [Reinforcement Learning](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/) | Reinforcement learning workflows and examples |
| [Troubleshooting](https://verl-vla.readthedocs.io/en/latest/troubleshooting/) | Guidance for diagnosing simulator and distributed execution problems |

## Contributing

We warmly welcome contributions. Valuable improvements of any kind, as well as
meaningful and reproducible experiments, can help more people bring embodied
models into real-world applications. If you encounter a problem or have an
idea for a new model, environment, device, training workflow, or experiment,
please open a
[GitHub issue](https://github.com/verl-project/verl-vla/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Acknowledgements

verl-vla is built on [verl](https://github.com/verl-project/verl), extending
its distributed training infrastructure for robotics and VLA post-training.
We sincerely thank the verl team for their foundational work and continued
support for this project.

We are grateful to [LeRobot](https://github.com/huggingface/lerobot),
[SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL),
[RLinf](https://github.com/RLinf/RLinf),
[DSRL](https://github.com/ajwagen/dsrl),
[Giga Models](https://github.com/open-gigaai/giga-models),
[OpenPI](https://github.com/Physical-Intelligence/openpi), and
[Evo-RL](https://github.com/MINT-SJTU/Evo-RL) for the ideas, implementations,
and open-source foundations that helped shape this project. In particular,
verl-vla's user-facing data and device APIs are organized with reference to
LeRobot's elegant API design.

We owe special thanks to the
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena) and
[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) projects and teams,
whose substantial contributions and close support have been instrumental to
verl-vla.

## License

verl-vla is licensed under the [Apache License 2.0](LICENSE).
