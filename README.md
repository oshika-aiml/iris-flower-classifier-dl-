# Iris Flower Classification (Deep Learning)

A neural network (ANN) built with TensorFlow/Keras that classifies iris flowers into 3 species (setosa, versicolor, virginica) from 4 measurements.

## Dataset
Iris dataset (150 samples, 50 per species), loaded from scikit-learn. No download needed.
Features: sepal length, sepal width, petal length, petal width (cm).

- Train: 120 samples
- Test: 30 samples

## Model
Input (4) -> Dense(16, ReLU) -> Dense(16, ReLU) -> Dropout(0.2) -> Dense(3, Softmax)

- Features scaled with StandardScaler
- Optimizer: Adam, loss: sparse categorical crossentropy
- 100 epochs, batch size 8
- Total parameters: 403

## Results

| Metric | Value |
|---|---|
| Train accuracy | 0.958 |
| Test accuracy | 0.933 |

Setosa was classified perfectly. The only errors were one versicolor predicted as virginica and one virginica predicted as versicolor, which overlap in feature space.

## How to run
Open Google Colab, paste the code from iris_flower_dl.py and run it.
Or locally:

    pip install -r requirements.txt
    python iris_flower_dl.py

## Future improvements
- Cross-validation for a more reliable estimate on a small dataset
- Compare with classical ML models (SVM, Random Forest)
- Streamlit app to predict from user input
