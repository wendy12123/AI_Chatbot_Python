# AI Knowledge Base: Gradient Descent & Neural Network Training

## 1. Multi-class Classification

Unlike binary classification (Yes/No), multi-class (or multinomial) classification predicts the probability of an outcome belonging to one of several possible classes.

* **Output Layer Structure:** For $m$ possible classes, the neural network should have $m$ nodes in the output layer.
* **Probability Distribution:** The model calculates the likelihood for each class (e.g., predicting weather as Sunny, Cloudy, or Foggy).
* **Fully Connected Layer:** In these networks, every input node connects to every output node, necessitating Gradient Descent to estimate the vast number of weights ($w_{nm}$).

---

## 2. Gradient Descent (GD)

Gradient Descent is the optimization algorithm used to minimize the cost function (error) of a neural network by iteratively adjusting weights and biases.

### The Mechanism

* **Analogy:** Imagine standing on a hill in a thick fog; to find the bottom of the valley, you feel the slope of the ground and take a step in the direction where it descends most steeply.
* **Cost Function ($C$):** The measure of the error between the predicted output and the actual target.
* **Weights Adjustment:** Weights are updated in the opposite direction of the gradient to "descend" toward the minimum error.

---

## 3. Key Hyperparameters in Training

Hyperparameters are settings that control the learning process and must be defined before training begins.

### Learning Rate ($\eta$)

The learning rate determines the size of the "steps" the algorithm takes down the gradient.

* **Too High:** The algorithm may overshoot the minimum and fail to converge.
* **Too Low:** The algorithm will be very accurate but will take an extremely long time to train.

### Batch Size and Epochs

* **Epoch:** One complete pass of the entire training dataset through the neural network.
* **Batch Size:** The number of training examples utilized in one iteration to update the weights.

---

## 4. Optimizers

Optimizers are advanced algorithms based on Gradient Descent that adapt the learning process to be faster and more stable.

* **Adam Optimizer:** A popular choice that computes individual adaptive learning rates for different parameters. It is generally faster and requires less manual tuning than standard Stochastic Gradient Descent (SGD).

---

## 5. Practical Implementation: House Price Prediction

The following steps outline the workflow for a regression task using an Artificial Neural Network (ANN).

### Data Preprocessing

* **Feature Selection:** Identifying relevant inputs (e.g., number of bedrooms, square footage).
* **Feature Scaling (Normalization):** Scaling data to a specific range (usually 0 to 1) using tools like `MinMaxScaler`. This prevents features with large numerical values from dominating the learning process.

### Model Building (Keras/TensorFlow)

1. **Initialize:** Create a Sequential model.
2. **Add Layers:** Define the input layer and hidden layers (typically using **ReLU** activation).
3. **Compile:** Choose the optimizer (e.g., 'adam') and the loss function (e.g., 'mean_squared_error' for regression).
4. **Fit:** Train the model using the training data, specifying the number of epochs.

### Evaluation

* **Predictions:** Pass new data through the trained model.
* **Error Rate:** Calculate the difference between predicted prices and actual prices to determine model accuracy.
