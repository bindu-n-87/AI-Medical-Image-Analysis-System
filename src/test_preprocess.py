from preprocess import load_data, normalize_data

# Load dataset
train, val, test = load_data()

# Normalize dataset
train, val, test = normalize_data(train, val, test)

# Check one batch
for images, labels in train.take(1):
    print("Image batch shape:", images.shape)
    print("Labels shape:", labels.shape)