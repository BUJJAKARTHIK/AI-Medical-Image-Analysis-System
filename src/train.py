from preprocess import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X, y = load_data("data/train")

# Normalize
X = X / 255.0

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Build model
model = build_model()

# Train model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Save model
model.save("models/model.h5")

# -------------------------
# 📊 PLOT GRAPH
# -------------------------

# Accuracy graph
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("outputs/accuracy.png")
plt.show()

# Loss graph
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("outputs/loss.png")
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Predictions on validation set
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()