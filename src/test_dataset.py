import os

train_path = "data/chest_xray/train"

for category in os.listdir(train_path):
    path = os.path.join(train_path, category)
    print(f"{category}: {len(os.listdir(path))} images")