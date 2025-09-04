# Probability & Statistics for Machine Learning & Data Science

This repository contains a collection of assignments and projects designed to explore and apply fundamental concepts of probability and statistics in the context of machine learning and data science. Through hands-on exercises, we delve into statistical analysis, hypothesis testing, and building predictive models using popular Python libraries like NumPy, Pandas, and TensorFlow.

## Concepts Covered

*   **Descriptive Statistics:** Calculating and interpreting measures like mean, variance, and standard deviation.
*   **Hypothesis Testing:**
    *   Conducting A/B tests to compare different versions of a product.
    *   Using t-tests to determine the statistical significance of differences between groups.
    *   Understanding and using p-values to make data-driven decisions.
*   **Correlation Analysis:** Measuring the relationship between different variables using correlation coefficients.
*   **Machine Learning with TensorFlow:**
    *   **Data Pipelines:** Creating efficient training and validation data pipelines from image directories using `tf.data.Dataset`.
    *   **Data Preprocessing:** Normalizing and reshaping image data for model consumption.
    *   **Convolutional Neural Networks (CNNs):** Designing, building, and compiling CNNs for image classification tasks.
    *   **Model Training & Evaluation:** Training models, monitoring performance metrics (accuracy, loss), and using callbacks for custom training logic like early stopping.

## Assignments

This repository includes several hands-on assignments:

1.  **Statistical Analysis with NumPy:** Analyzing gene expression data to understand the role of specific genes in biological processes. This involves calculating mean differences, implementing t-tests, and handling missing data.
2.  **A/B Testing for Product Improvement:** Performing hypothesis tests for both continuous (e.g., average session duration) and proportion-based metrics to make decisions about product changes.
3.  **Image Classification with TensorFlow:** Building, training, and evaluating a Convolutional Neural Network (CNN) to classify images of cats and dogs.

## Example Project: Cats vs. Dogs Image Classification

This project demonstrates an end-to-end machine learning workflow for a binary image classification problem.

### 1. Data Pipeline

We use `tf.keras.utils.image_dataset_from_directory` to create training and validation datasets directly from the image folders. This function handles loading images, resizing them, creating batches, and splitting the data.

```python
def train_val_datasets():
    """Creates datasets for training and validation."""
    
    # Directory that holds the data
    DATA_DIR = '/tmp/PetImages'

    # Create the training dataset
    training_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=(150, 150),
        batch_size=128,
        label_mode='binary',
        validation_split=0.1,
        subset='training',
        seed=42
    )

    # Create the validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=(150, 150),
        batch_size=128,
        label_mode='binary',
        validation_split=0.1,
        subset='validation',
        seed=42
    )

    return training_dataset, validation_dataset
```

### 2. Model Architecture

A sequential CNN model is created with three convolutional blocks followed by dense layers for classification. Pixel values are rescaled to the `[0, 1]` range as the first step.

```python
def create_model():
    """Creates the untrained model for classifying cats and dogs."""
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]) 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model
```

### 3. Custom Callback for Early Stopping

A custom `EarlyStoppingCallback` is implemented to monitor training and validation accuracy, stopping the training process once the desired performance is achieved to prevent overfitting and save computational resources.

```python
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.95 and logs.get('val_accuracy') >= 0.8:
            print("\nReached 95% train accuracy and 80% validation accuracy, so cancelling training!")
            self.model.stop_training = True
```

