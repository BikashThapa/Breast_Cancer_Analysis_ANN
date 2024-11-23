# Project Topic: 
Breast Cancer Analysis Using ANN
It is also available on the UCI Machine Learning Repository: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

Additionally, the dataset can be accessed via Kaggle: [Breast Cancer Wisconsin Data (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

### Attribute Information:

| **Attribute**                 | **Description**                                                                                                                                                                  |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1) ID number**               | Unique identifier for each record.                                                                                                                                               |
| **2) Diagnosis**               | Class label: M = Malignant, B = Benign                                                                                                                                             |
| **3-32) Features**             | Ten real-valued features computed for each cell nucleus (mean, standard error, and worst).                                                                                       |
| **a) Radius**                  | Mean of distances from center to points on the perimeter.                                                                                                                       |
| **b) Texture**                 | Standard deviation of gray-scale values.                                                                                                                                         |
| **c) Perimeter**               | Perimeter of the nucleus.                                                                                                                                                       |
| **d) Area**                    | Area of the nucleus.                                                                                                                                                           |
| **e) Smoothness**              | Local variation in radius lengths.                                                                                                                                               |
| **f) Compactness**             | Perimeter^2 / area - 1.0                                                                                                                                                        |
| **g) Concavity**               | Severity of concave portions of the contour.                                                                                                                                     |
| **h) Concave Points**          | Number of concave portions of the contour.                                                                                                                                       |
| **i) Symmetry**                | Symmetry of the nucleus.                                                                                                                                                        |
| **j) Fractal Dimension**       | "Coastline approximation" - 1.                                                                                                                                                  |

### Feature Calculation:
For each of the above 10 features, the following values were computed:
- **Mean**: The average of the values for each feature.
- **Standard Error**: The standard error of the feature.
- **Worst**: The largest value of the feature (mean of the three largest values).

These calculations result in **30 features** (3 for each of the 10 original features), which are used for classification.

### Class Distribution:
- **Benign**: 357 instances
- **Malignant**: 212 instances

### Missing Attribute Values:
- **None**

## Steps Followed:

### 1. **Exploratory Data Analysis (EDA):**
   - The dataset was analyzed to understand its structure and identify patterns in the features.
   - Visualizations were created to better understand the distribution of features and their relationships with the target variable (malignant or benign).
   - Missing data, correlations, and feature distributions were explored.

### 2. **Data Cleaning:**
   - Missing or inconsistent data was handled.
   - The target variable `diagnosis` was encoded (Benign = 0, Malignant = 1).
   - Any duplicates or irrelevant information were removed from the dataset.

### 3. **Preprocessing and Feature Selection:**
   - **Standardization**: We standardized the data using `StandardScaler` to ensure features are on the same scale, as the ANN model (MLPClassifier) is sensitive to the scale of input features.
   - **Feature Selection**: We used **ANOVA F-value** and **Mutual Information** to select the most important features (about 10) that contribute to the prediction of the target variable. This helps reduce model complexity and improve generalization.
   - The selected features included measurements related to the tumor's radius, perimeter, area, and concavity, which are known to be relevant in diagnosing breast cancer.

### 4. **Model Building:**
   - The selected features were used to train a **Multi-layer Perceptron (MLP) classifier**, which is a type of Artificial Neural Network (ANN).
   - The model architecture used two hidden layers, each with 50 neurons.
   - The model was trained with the training data and evaluated using accuracy, precision, recall, and F1-score.

### 5. **Model Tuning:**
   - **Grid Search Cross-validation**: We optimized the model's hyperparameters using Grid Search, selecting values for parameters like `activation`, `alpha`, `batch_size`, and `hidden_layer_sizes`.
   - The best parameters were found to be:
     - `activation`: 'relu'
     - `alpha`: 0.0001
     - `batch_size`: 64
     - `hidden_layer_sizes`: (100,)
     - `learning_rate`: 'constant'
     - `solver`: 'adam'
   - These tuned parameters were used to train the final model, resulting in improved performance.

### 6. **Model Evaluation:**
   - The final model achieved an accuracy of **97%** on the test set with a precision of 0.97 for benign cases and 0.98 for malignant cases.
   - The classification report and confusion matrix indicated that the model performed well, with high recall (0.99) for benign cases and good overall F1-scores.
   - **Confusion Matrix**:
     ```
     [[70  1]
      [ 2 41]]
     ```

### 7. **Deployment:**
   - The trained model (`mlp_model_imp.pkl`) and the scaler (`scaler.pkl`) were saved using `pickle`.
   - The model and scaler can be loaded into a **Streamlit app**, allowing users to input feature values and predict whether a tumor is benign or malignant.

## Conclusion:
This project demonstrates how machine learning can be effectively applied to the classification of breast cancer tumors. By preprocessing the data, selecting important features, and tuning the model, we were able to achieve high accuracy and build a robust system for detecting malignant tumors. The model is now ready for deployment and can be used in real-world applications to assist in breast cancer diagnosis.
