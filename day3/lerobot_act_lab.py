import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


# ============================================================
# 1. Policy / dataset temporal configuration
# ============================================================

FPS = 10.0
CHUNK_SIZE = 100
BATCH_SIZE = 4

delta_timestamps = {
    "action": [i / FPS for i in range(CHUNK_SIZE)],
}


# ============================================================
# 2. Dataset
#
# Each sample contains:
#
#   observation.image : [3, 96, 96]
#   observation.state : [2]
#   action            : [CHUNK_SIZE, 2]
#
# delta_timestamps converts the original single-step action
# sequence into an action chunk for training.
# ============================================================

dataset = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps=delta_timestamps,
)

print("=== Dataset ===")
print("num samples:", len(dataset))

sample = dataset[0]

print("image :", sample["observation.image"].shape)
print("state :", sample["observation.state"].shape)
print("action:", sample["action"].shape)


# ============================================================
# 3. Dataset features -> Policy features
#
# Dataset metadata:
#   video / float32 / fps / ...
#
# becomes:
#   VISUAL / STATE / ACTION
# ============================================================

features = dataset_to_policy_features(dataset.meta.features)

print("\n=== Policy features ===")

for name, feature in features.items():
    print(name, feature)


# ============================================================
# 4. ACT configuration
#
# observation.image -> ResNet18
# observation.state -> state projection
#
#                  ↓
#
#              Transformer
#
#                  ↓
#
#       action chunk [100, 2]
# ============================================================

config = ACTConfig(
    input_features={
        "observation.image": features["observation.image"],
        "observation.state": features["observation.state"],
    },
    output_features={
        "action": features["action"],
    },
    chunk_size=CHUNK_SIZE,
    n_action_steps=CHUNK_SIZE,
    device="cuda",
)

policy = ACTPolicy(config).to("cuda")

print("\n=== ACT ===")
print("device:", next(policy.parameters()).device)
print("chunk_size:", config.chunk_size)


# ============================================================
# 5. DataLoader
#
# Single sample:
#
#   image  [3, 96, 96]
#   state  [2]
#   action [100, 2]
#
# Batch:
#
#   image  [B, 3, 96, 96]
#   state  [B, 2]
#   action [B, 100, 2]
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

batch = next(iter(dataloader))

print("\n=== Batch ===")
print("image :", batch["observation.image"].shape)
print("state :", batch["observation.state"].shape)
print("action:", batch["action"].shape)


# ============================================================
# 6. Move tensor inputs to GPU
# ============================================================

batch = {
    key: value.to("cuda") if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}


# ============================================================
# 7. One training forward pass
#
# ACT training uses the expert action chunk as well.
#
# Conceptually:
#
# expert action chunk ---> VAE encoder ---> latent z
#                                          |
# image ---> ResNet18 ---------------------+
# state -----------------------------------+
#                                          |
#                                   Transformer
#                                          |
#                                          v
#                              predicted action chunk
#
# Loss includes action reconstruction + KL loss.
# ============================================================

policy.train()

loss, loss_dict = policy(batch)

print("\n=== Forward ===")
print("loss:", loss.item())

for name, value in loss_dict.items():
    if isinstance(value, torch.Tensor):
        value = value.item()

    print(f"{name}: {value}")


# ============================================================
# 8. One backward pass
#
# Same fundamental PyTorch mechanism as our tiny BC example:
#
#   prediction
#       ↓
#     loss
#       ↓
#   backward()
#
# The policy itself is much more sophisticated, but autograd
# and optimization are still the same mechanism.
# ============================================================

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=config.optimizer_lr,
    weight_decay=config.optimizer_weight_decay,
)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nOne ACT training step completed.")
