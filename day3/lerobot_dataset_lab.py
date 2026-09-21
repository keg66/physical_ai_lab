from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps={
        "action": [0.0, 0.1, 0.2, 0.3],
    },
)

print(dataset)
print("len =", len(dataset))

sample = dataset[0]

print("\n====sample====")
print(type(sample))
print(sample.keys())

print("\n====key, value====")
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(key, value.shape, value.dtype)
    else:
        print(key, type(value), value)

print("====first 10 frames====")
first_frame = dataset[0]
episode_index = first_frame["episode_index"]

print("frame_index\ttimestamp\tobservation.state\taction")

shown = 0
dataset_index = 0
while shown < 10 and dataset_index < len(dataset):
    frame = dataset[dataset_index]
    if frame["episode_index"] != episode_index:
        break

    print(
        frame["frame_index"],
        frame["timestamp"],
        frame["observation.state"],
        frame["action"],
        sep="\t",
    )
    shown += 1
    dataset_index += 1
