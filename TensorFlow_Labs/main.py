import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load dataset
data = pd.read_csv('E:/development/deepLearning/diabetes.csv')
X = data[['Glucose', 'Insulin']].values
y = data['Outcome'].values

# Normalize features using TensorFlow layer (same as your approach)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X)
X_norm = normalizer(X)

# Convert TensorFlow tensor to NumPy arrays
X_norm = X_norm.numpy()  # Now it's a NumPy array

# Split data manually (80% train, 20% test)
train_size = int(0.8 * X.shape[0])
X_train, X_test = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Example: Undersample negatives
pos_indices = np.where(y_train == 1)[0]
neg_indices = np.where(y_train == 0)[0][:len(pos_indices)]

balanced_indices = np.concatenate([pos_indices, neg_indices])
X_train = X_train[balanced_indices]
y_train = y_train[balanced_indices]


# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function (used for backpropagation)
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(1. / hidden_size)
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, A2

# Compute cost
def compute_cost(A2, y):
    m = y.shape[0]
    cost = -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
    return cost

# Backpropagation
def backpropagation(X, y, A1, A2, W1, W2):
    m = X.shape[0]
    
    # Compute gradients
    dZ2 = A2 - y.reshape(-1, 1)  # shape (m, 1)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Train the model
def train_model(X_train, y_train, hidden_size=3, epochs=1000, learning_rate=0.01):
    input_size = X_train.shape[1]
    output_size = 1
    
    # Initialize parameters
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        A1, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        
        # Compute cost
        cost = compute_cost(A2, y_train)
        
        # Backpropagation
        dW1, db1, dW2, db2 = backpropagation(X_train, y_train, A1, A2, W1, W2)
        
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        # Print the cost every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return W1, b1, W2, b2

# Predict using the trained model
def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return A2

# Train the model from scratch
W1, b1, W2, b2 = train_model(X_train, y_train, hidden_size=5, epochs=1000, learning_rate=0.02)

# Predict on test data
predictions = predict(X_test, W1, b1, W2, b2)
predictions = (predictions >= 0.5).astype(int)  # Convert probabilities to binary labels

# Print predictions and actual values
print("\nPredictions on Test Set:")
for i in range(len(y_test)):
    print(f"Predicted: {predictions[i][0]}, Actual: {y_test[i]}")

# Visualize the predictions
colors = ['green' if label == 1 else 'red' for label in predictions.ravel()]

plt.figure(figsize=(8, 6))
for i in range(len(X_test)):
    plt.scatter(X_test[i, 0], X_test[i, 1], color=colors[i], label='Positive' if predictions[i] == 1 else 'Negative')

plt.xlabel('Glucose')
plt.ylabel('Insulin')
plt.title('Test Data Predictions')
plt.legend(loc='upper right')
plt.grid(True)
accuracy = np.mean(predictions.flatten() == y_test)
print(f"Test Accuracy: {accuracy:.2f}")
plt.show()
