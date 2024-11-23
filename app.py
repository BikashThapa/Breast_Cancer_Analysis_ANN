
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
# Load the pre-trained model and scaler
with open(r'G:\Projects\Breast-Cancer-Data-Analysis\Pickle-Files\mlp_model_imp.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'G:\Projects\Breast-Cancer-Data-Analysis\Pickle-Files\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define all 30 feature names (complete list of features used in training)
all_feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Streamlit UI
st.title("Breast Cancer Prediction using ANN Model")
st.write("This app uses a pre-trained Artificial Neural Network (ANN) model to predict the diagnosis of breast cancer (Benign or Malignant).")

# User inputs for the features (using sliders for each of the 30 features)
feature_values = {}
for feature in all_feature_names:
    feature_values[feature] = st.slider(f"Select {feature}", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Convert the user inputs to a numpy array (to pass into the model)
user_input = np.array(list(feature_values.values())).reshape(1, -1)

# Scale the input using the pre-trained scaler
user_input_scaled = scaler.transform(user_input)

# Model Prediction
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)

    # Display the result
    if prediction == 0:
        st.subheader("Prediction: Benign (B)")
    else:
        st.subheader("Prediction: Malignant (M)")

    st.write("### Feature Values Entered:")
    for feature, value in feature_values.items():
        st.write(f"{feature}: {value}")



