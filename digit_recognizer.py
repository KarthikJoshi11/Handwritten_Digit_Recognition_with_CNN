import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset. It is already split into training and testing sets.
# The data consists of 28x28 grayscale images of handwritten digits.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data Preprocessing
# Normalize the pixel values from [0, 255] to [0, 1] to help the model learn faster.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to add a channel dimension.
# CNNs expect a 4D input: (number of samples, height, width, channels).
# For grayscale images, the channel is 1.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Define the CNN Model Architecture
model = keras.Sequential([
    # First Convolutional Layer
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max Pooling Layer to downsample the feature map
    keras.layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Layer
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Another Max Pooling Layer
    keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten the 3D output to 1D to feed into a Dense (fully connected) layer
    keras.layers.Flatten(),
    
    # Fully Connected Layer
    keras.layers.Dense(64, activation='relu'),
    
    # Output Layer with 10 neurons (for digits 0-9) and softmax activation
    # Softmax ensures the output probabilities sum to 1.
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Optimizer: 'adam' is a good default choice.
# Loss: 'sparse_categorical_crossentropy' is used for multi-class classification when labels are integers.
# Metrics: 'accuracy' is used to monitor performance.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# epochs=10 means the model will see the entire training data 10 times.
print("Training the model...")
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the test data
print("\nEvaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Make predictions on a few test images
predictions = model.predict(x_test)

# Let's visualize the first 5 test images and their predicted labels
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Display the image
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    # Get the predicted label (the one with the highest probability)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    # Color the title red if the prediction is wrong, green if it's correct
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f"Predicted: {predicted_label} (True: {true_label})", color=color)
plt.show()

# Print the model summary to see its structure and parameter count
model.summary()
