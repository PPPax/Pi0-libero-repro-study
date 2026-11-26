# Pi0-libero-repro-study

Single-RTX-5090 reproduction of Dexbotic π0, including two-task LIBERO-Spatial LoRA fine-tuning, closed-loop evaluation, and small-scale action-expert PPO post-training.

## Project Overview

Hardware specifications and environment dependencies are listed below.

### Hardware

- GPU: NVIDIA GeForce RTX 5090
- GPU memory: 32,607 MiB; compute capability: 12.0 (sm_120)
- System memory: approximately 31 GiB

### Environment

- Docker image: `dexmal/dexbotic:c130t28`
- Python: 3.12.3
- CUDA runtime: 13.0
- Model implementation: Dexbotic π0
- Inference and LoRA fine-tuning use the model container; LIBERO evaluation and PPO use dedicated project environments.

### Core Dependencies

- `torch==2.8.0a0+34c6371d24.nv25.8`
- `torchvision==0.23.0a0+428a54c9`
- `transformers==4.54.0`, `tokenizers==0.21.4`
- `peft==0.18.0`, `accelerate==1.12.0`
- `timm==1.0.24`, `numpy==1.26.4`
- `safetensors==0.6.2`, `huggingface-hub==0.36.0`
- `wandb==0.23.1`
- `ray==2.47.0` in the PPO environment

### LIBERO Simulation Dependencies

- LIBERO: source checkout
- `robosuite==1.4.1`
- `mujoco==3.2.3`
- `bddl==1.0.1`
- EGL for headless GPU rendering

This setup is used to deploy Dexbotic π0 and support inference, LoRA fine-tuning, LIBERO evaluation, and small-scale PPO post-training. For setup details and troubleshooting notes, see the [pi0_notes](https://blog.csdn.net/ccccc121212wtf/article/details/170191460?spm=1001.2014.3001.5502).


## LoRA Fine-tuning

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | Dexmal/Dexbotic-PI0 |
| Dataset | π0-specific LIBERO-Spatial data, task IDs 0 and 1 |
| Demonstrations / frames | 90 / 10,427 |
| Micro batch size | 1 |
| Gradient accumulation steps | 16 |
| Effective batch size | 16 |
| LoRA rank / alpha / dropout | 32 / 16 / 0 |
| LoRA targets | `all-linear`, with selected projection modules fully trained |
| Training steps | 8,000 optimizer updates |
| Learning rate | 500-step warmup to 5e-4, followed by cosine decay to approximately 5e-5 |
| Precision | BF16 computation with FP32 parameters |
| Checkpoint interval | Every 500 optimizer updates; all checkpoints retained |
| Logging | Weights & Biases |

### Training Curves

**Train Loss:**
<img width="1390" height="877" alt="image" src="https://github.com/user-attachments/assets/663df4d4-2c3b-4c29-a1de-85aeea1dea86" />


The flow-matching loss dropped rapidly at the beginning and then decreased more gradually, with occasional spikes. The first and final 100-step averages fell from 0.1525 to 0.00993, indicating improved fitting of the training objective; closed-loop performance was evaluated separately.


**Gradient Norm:**
<img width="1390" height="877" alt="image" src="https://github.com/user-attachments/assets/51bea89e-e549-4b55-af01-819c9a449e9d" />



The gradient norm showed an early spike and then decreased overall. Its first and final 100-step averages fell from approximately 1.025 to 0.127, showing smaller gradient magnitudes later in training without establishing improved task success.

**GPU Memory Used:**
<img width="1366" height="829" alt="image" src="https://github.com/user-attachments/assets/2fb18650-c601-4665-8ef5-170ab25d1f53" />


Sampled GPU memory usage ranged from approximately 27.62 to 28.10 GiB. After the initial fluctuations, it remained near 27.75 GiB with small recurring changes, showing broadly stable sampled memory usage during the later training period.




### Evaluation Protocol

Each model was evaluated on task IDs 0 and 1 using 10 fixed initial states per task: 20 episodes per model.

- Predict 50 actions and execute the first 5 before replanning.
- Keep the training action horizon at 50.
- Use the same normalization statistics and evaluation settings.
- Allow up to 220 control steps per episode.

| Model | Task 0 | Task 1 | Overall Success Rate |
|---|---:|---:|---:|
| INIT | 0/10 | 0/10 | 0% |
| checkpoint-6000 | 9/10 | 2/10 | 55% |
| checkpoint-6500 | 8/10 | 5/10 | 65% |
| **checkpoint-7000** | **10/10** | **8/10** | **90%** |
| checkpoint-7500 | 9/10 | 6/10 | 75% |
| checkpoint-8000 | 10/10 | 7/10 | 85% |

Closed-loop performance did not improve monotonically with training steps. Checkpoint-7000 achieved the best result under this protocol and was selected for PPO initialization. 



### Successful Episode Demos

#### Task 0: Bowl Between the Plate and the Ramekin




https://github.com/user-attachments/assets/1075bd3d-190f-4ec0-9b28-8597638c7163




#### Task 1: Bowl Next to the Ramekin





https://github.com/user-attachments/assets/c0e1fbab-5db9-4569-8f8a-d5f2d1478bdf






### Failure Analysis

Across the five SFT checkpoints, 26 of 100 episodes failed. All failed episodes reached the 220-step limit without recorded program errors.

| Observed Failure Category | Episodes |
|---|---:|
| No effective target grasp | 11 |
| Off-center placement after transport | 12 |
| Object dropped outside the plate | 2 |
| Unstable grasp/contact | 1 |

The failed SFT episodes did not show whole-episode immobility. Recorded cumulative end-effector travel ranged from approximately 656.6 to 2,392.9 mm. In contrast, all 20 INIT episodes failed to establish a valid target grasp and showed motion away from the intended tabletop manipulation.


These observations identify failure stages, but do not uniquely establish causes such as grip force, friction, or training-data coverage.


Using the best-performing checkpoint-7000 as a case study, I examine its two failed rollouts: task ID 1, episode indices 4 and 5.
<img width="960" height="476" alt="checkpoint-7000_failures_00" src="https://github.com/user-attachments/assets/467eb733-1a06-435a-acd9-bf90aa2bff68" />

The frames show grasping and transport rather than whole-episode immobility. Although the final placements appear close to successful, replay measurements showed bowl-to-plate XY center offsets of 36.33 and 47.65 mm. Both exceeded the native success predicate's 30 mm distance threshold, despite final plate contact and the bowl being above the plate.

#### Failed Episode Demos




https://github.com/user-attachments/assets/e96e6b65-a78e-4099-b8c7-d069b0e50b7f




https://github.com/user-attachments/assets/21162f3c-d253-4552-97bb-d8edee893f21






## Action Expert PPO Post-training

Within a single RTX 5090 and the available host-memory budget, this stage focused on feasibility validation and short PPO training runs, including real parameter updates and checkpoint recovery. No reliable improvement over SFT was established. For the detailed experimental procedure and troubleshooting notes, see the [pi0_notes](https://blog.csdn.net/ccccc121212wtf/article/details/170191460?spm=1001.2014.3001.5502).

### Training Configuration

| Parameter | Value |
|---|---|
| Initialization | Fully merged SFT checkpoint-7000 |
| Frozen modules | SFT-updated VLM |
| Trainable modules | Full action expert, associated projections, and value head |
| Trainable parameters | 314,714,145 |
| Actor / value learning rate | 5e-6 / 1e-4 |
| Update epochs per rollout batch | 1 |
| Sampling | One environment, one episode per outer iteration |
| Batch configuration | 44 decision slots, micro batch 1, gradient accumulation 44 |
| Optimizer updates | One per outer iteration |
| Policy clip / value clip | 0.2 / 0.2 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Advantage normalization | Enabled |
| Gradient clipping | Global norm limit of 1.0 |
| Action sampling | Flow-SDE, noise level 0.5, 10 denoising steps |
| Action execution | Predict 50 actions; execute 5 before replanning |

### Training Curves
<img width="2400" height="1344" alt="ppo_training_curves" src="https://github.com/user-attachments/assets/618a4394-512e-4293-9643-0f151ccc011b" />


Versions 1–3 are preliminary validation; versions 4–23 are the formal continuation. The horizontal axis records the accepted version after each update. Episode outcomes come from that iteration's pre-update rollout, rather than separate checkpoint evaluations.

- **A — Episode outcomes:** 6 of 20 formal training episodes succeeded. The first 10 contained 5 successes; the last 10 contained 1. Early rolling-window values also include preliminary episodes.
- **B — Value loss:** Lower prediction error did not consistently correspond to better task performance.
- **C — Explained variance:** Mostly negative values indicated unstable value prediction; low target variance can amplify this metric.
- **D — Gradient norm:** Values are recorded before clipping. Exceeding 1.0 does not indicate clipping failure.
- **E — Valid decisions:** Masks determined how many of the 44 slots contributed to learning. The formal segment contained 755 valid decisions across 880 slots.
- **F — Policy loss:** Near-zero values were consistent with normalized advantages and a single full-batch update. They do not imply zero gradients or policy convergence.

### Results and Limitations

The formal segment continued from policy version 3 to 23, accepting 20 new optimizer updates. Checkpoints at versions 8, 13, 18, and 23 were saved and reload-verified.

| Measurement | Result |
|---|---:|
| Formal training rollouts | 6/20 successful |
| SFT7000, four matched conditions | 3/4 successful |
| PPO23, the same four conditions | 1/4 successful |
| Observed peak GPU memory | Approximately 26.50 GiB |
| Highest sampled GPU temperature | 63°C |

Training-rollout outcomes and matched-condition evaluations are different measurements and should not be directly combined.

The staged single-GPU workflow completed real PPO updates, weight synchronization, checkpoint saving, and recovery. However, the small matched-condition comparison showed a decline, and no reliable improvement over SFT was established.

## Acknowledgements

This project builds on:

- [Dexbotic](https://github.com/Dexmal/dexbotic)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [RLinf](https://github.com/RLinf/RLinf)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
