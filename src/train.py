import matplotlib.pyplot as plt
from preprocess import load_data, normalize_data
from model import build_model

train_data, val_data, test_data = load_data()
train_data, val_data, test_data = normalize_data(train_data, val_data, test_data)

model = build_model()

EPOCHS = 5

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save("models/medical_model.h5")

print("Model saved successfully!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 5))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.savefig("outputs/training_results.png")
plt.show()
