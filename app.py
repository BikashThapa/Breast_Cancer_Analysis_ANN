import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
with open(r'G:\Projects\Breast-Cancer-Data-Analysis\Pickle-Files\mlp_model_imp.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'G:\Projects\Breast-Cancer-Data-Analysis\Pickle-Files\scaler_imp.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Dictionary for min and max values of the selected features
feature_min_max = {
    'radius_mean': (6.981, 28.11),
    'perimeter_mean': (43.79, 188.5),
    'area_mean': (143.5, 2501.0),
    'concavity_mean': (0.0, 0.4268),
    'concave points_mean': (0.0, 0.2012),
    'radius_worst': (7.93, 36.04),
    'perimeter_worst': (50.41, 251.2),
    'area_worst': (185.2, 4254.0),
    'concavity_worst': (0.0, 1.252),
    'concave points_worst': (0.0, 0.291)
}

# User inputs for the selected features (10 features)
selected_feature_names = list(feature_min_max.keys())

# Create a layout with 2 columns: one for sliders (left), one for prediction (right)
col1, col2 = st.columns([2, 3])  # The first column will be 2 parts, second will be 3 parts

with col1:
    st.header("Feature Inputs")
    feature_values = {}
    for feature in selected_feature_names:
        # Get the min and max values from the feature_min_max dictionary
        min_val, max_val = feature_min_max[feature]
        feature_values[feature] = st.slider(
            f"Select {feature}",
            min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=0.1)

with col2:
    st.header("Prediction Result")

    # Convert the user inputs to a numpy array (to pass into the model)
    user_input = np.array([feature_values[feature] for feature in selected_feature_names]).reshape(1, -1)

    # Scale the input using the pre-trained scaler
    user_input_scaled = scaler.transform(user_input)

    # Make the prediction using the loaded model
    if st.button("Predict"):
        prediction = model.predict(user_input_scaled)

        # Display the result
        if prediction == 0:
            prediction_label = "Benign (B)"
            explanation = """
            This means the tumor is non-cancerous. The model found that the features provided 
            (e.g., smaller size, less irregularity) match the patterns typically seen in benign tumors.
            """
        else:
            prediction_label = "Malignant (M)"
            explanation = """
            This means the tumor is cancerous. The model identified characteristics such as larger size 
            and more irregularity in shape, which are common traits of malignant tumors.
            """

        # Show prediction and explanation
        st.subheader(f"Prediction: {prediction_label}")
        st.write(explanation)

        # Show the features entered by the user
        st.write("### Feature Values Entered:")
        for feature, value in feature_values.items():
            st.write(f"{feature}: {value}")