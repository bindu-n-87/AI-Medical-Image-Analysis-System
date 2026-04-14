import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import load_data, normalize_data
from tensorflow.keras.models import load_model

train, val, test = load_data()
train, val, test = normalize_data(train, val, test)

model = load_model("models/medical_model.h5")

y_true = []
y_pred = []

for images, labels in test:
    preds = model.predict(images)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print("\nFinal Test Accuracy:", accuracy)
