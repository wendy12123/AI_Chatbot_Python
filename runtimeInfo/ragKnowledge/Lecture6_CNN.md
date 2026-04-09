# AI Knowledge Base: Convolutional Neural Networks (CNNs)

## 1. Computer Vision Fundamentals

Computer vision allows machines to "see" by interpreting images as numerical data.

* **Pixel Representation:** Images are composed of pixels. In grayscale, values range from 0 (black/off) to 255 (white/full brightness).
* **Color Channels:** Colored images are typically represented by three matrices corresponding to Red, Green, and Blue (RGB) values.
* **The Challenge:** Traditional neural networks struggle with images because high-resolution photos contain too many pixels, leading to an unmanageable number of weights. CNNs solve this by extracting the most important features.

---

## 2. CNN Architecture Components

A CNN processes images through a series of specialized layers to reduce dimensionality while preserving spatial relationships.

[Image of Convolutional Neural Network architecture layers]

### Convolutional Layer

* **The Process:** A small matrix called a **Kernel (or Filter)** slides across the input image.
* **Element-wise Multiplication:** The kernel multiplies its values with the pixel values it covers and sums them up.
* **Feature Map:** The result of this operation is a "Feature Map" (or Activation Map) that highlights specific patterns like edges, curves, or textures.

### Activation Function (ReLU)

* After convolution, the **Rectified Linear Unit (ReLU)** is applied to increase non-linearity in the image, as images themselves are highly non-linear. It replaces all negative pixel values in the feature map with zero.

### Pooling Layer

* **Purpose:** Down-sampling the feature map to reduce its size and the number of parameters the network needs to learn.
* **Max Pooling:** A common technique where a window slides over the feature map and only the maximum value in that window is kept. This makes the network robust to small distortions or shifts in the image.

### Flattening

* Once the image has been processed through convolution and pooling layers, the resulting 2D matrices are "flattened" into a long 1D vector. This vector serves as the input for a standard fully-connected neural network.

---

## 3. The Full CNN Workflow

1. **Input Image:** The raw pixel data is fed into the system.
2. **Feature Extraction:** Multiple pairs of **Convolution + Pooling** layers are chained together. Initial layers detect low-level features (edges), while deeper layers detect high-level features (shapes, objects).
3. **Fully Connected Layer:** The flattened features are passed through hidden layers (ANN) to perform the final classification.
4. **Output:** A **Softmax** function is usually applied at the end to provide a probability distribution across the possible categories (e.g., "Dog: 90%, Cat: 10%").

---

## 4. Key Advantages of CNNs

* **Shift Invariance:** Because of pooling, the network can recognize an object regardless of where it is located in the frame.
* **Parameter Efficiency:** Sharing weights through kernels allows the network to have far fewer parameters than a standard dense network, making it faster and easier to train on large image datasets.
