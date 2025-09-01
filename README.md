### Handwritten Digit Recognition with CNN

This repository contains a Python script that implements a Convolutional Neural Network (CNN) to recognize handwritten digits. The model is trained on the classic MNIST dataset, which is automatically downloaded as part of the script execution.

#### Key Features

  - **Convolutional Neural Network (CNN):** A deep learning model specifically designed for image classification tasks.
  - **Automatic Dataset Handling:** The script automatically downloads and loads the MNIST dataset, which includes 60,000 training images and 10,000 test images of handwritten digits.
  - **Data Preprocessing:** The code normalizes pixel values and reshapes the data to prepare it for the CNN.
  - **Performance Evaluation:** The model's accuracy is evaluated on a separate test set to measure its performance.
  - **Visualization:** The script generates a plot to show a few sample predictions, helping to visualize the model's performance.

#### Files in this Repository

  - `digit_recognizer.py`: The main script that defines, trains, and evaluates the CNN model.

#### How to Run the Code

1.  **Install the required libraries:**
    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    pip install tensorflow matplotlib numpy
    ```

2.  **Run the script:**

    ```bash
    python digit_recognizer.py
    ```

The script will handle the rest, including downloading the dataset, training the model, and displaying the test accuracy and a visualization of the predictions.
