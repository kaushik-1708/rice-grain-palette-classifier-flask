import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from prepare_data import validation_generator
from tensorflow.keras.models import load_model
model = load_model("rice_classifier_mobilenetv2.h5")


# Step 1: Evaluate model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
print(f"❌ Validation Loss: {val_loss:.4f}")

# Step 2: Predict
preds = model.predict(validation_generator)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes
labels = list(validation_generator.class_indices.keys())

# Step 3: Classification report
print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Step 4: Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Rice Type Classification - Confusion Matrix")
plt.show()
