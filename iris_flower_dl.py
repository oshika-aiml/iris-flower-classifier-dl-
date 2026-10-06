# Iris Flower Classification (Deep Learning)
# Neural network (ANN) built with TensorFlow/Keras
# Dataset: Iris flowers (150 samples, 3 species), built into scikit-learn - no download

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target
names = list(iris.target_names)
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [names[i] for i in y]
print(df.head())
print(df['species'].value_counts())
sns.pairplot(df, hue='species')
plt.show()

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 3. Build the neural network
model = models.Sequential([layers.Input(shape=(4,)), layers.Dense(16, activation='relu'), layers.Dense(16, activation='relu'), layers.Dropout(0.2), layers.Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)
print("Final train accuracy:", round(history.history['accuracy'][-1], 4))
print("Final validation accuracy:", round(history.history['val_accuracy'][-1], 4))

# 5. Training curves
plt.figure(figsize=(11,4))
plt.subplot(1,2,1); plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='validation'); plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
plt.subplot(1,2,2); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='validation'); plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.show()

# 6. Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", round(test_acc, 4))
preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, preds, target_names=names))
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
plt.title('Confusion Matrix - Iris ANN'); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

# 7. Predict a new flower (sepal length, sepal width, petal length, petal width in cm)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
print("Prediction for", sample[0].tolist(), "->", names[np.argmax(model.predict(sc.transform(sample), verbose=0))])

# 8. Save the model
model.save('iris_ann.keras')
print("Model saved as iris_ann.keras")
