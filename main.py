import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your training dataset
df_train = pd.read_csv('startup_ad_campaign_data.csv')

# Create a new binary column 'Conversion' based on some criteria
roas_threshold = 2.0
df_train['Conversion'] = (df_train['ROAS'] > roas_threshold).astype(int)

# Assume the new 'Conversion' column is the target variable, and the rest are features
X_train = df_train.drop('Conversion', axis=1)
y_train = df_train['Conversion']

# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical features
X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)

# Split the training dataset into training and testing sets
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define preprocessing steps (handle missing values and scale features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.columns)  # Scale numerical features if any
    ])

# Define the model (Logistic Regression for predicting Conversion)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Streamlit App
st.title("Ad Campaign Conversion Predictor")

# Create input widgets for user input
st.sidebar.header("User Input")

campaign_name = st.sidebar.text_input("Campaign Name", "New Campaign")
impressions = st.sidebar.number_input("Impressions", value=50000)
clicks = st.sidebar.number_input("Clicks", value=2000)
cost = st.sidebar.number_input("Cost", value=2000.0)
roas = st.sidebar.number_input("ROAS", value=3.0)
ad_creative_performance = st.sidebar.selectbox("Ad Creative Performance", ["High", "Medium", "Low"])
start_date = st.sidebar.text_input("Start Date", "2023-11-01")
end_date = st.sidebar.text_input("End Date", "2023-11-30")

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    'Campaign Name': [campaign_name],
    'Impressions': [impressions],
    'Clicks': [clicks],
    'Cost': [cost],
    'ROAS': [roas],
    'Ad Creative Performance': [ad_creative_performance],
    'Start Date': [start_date],
    'End Date': [end_date]
})

# One-hot encode categorical features
user_input = pd.get_dummies(user_input, columns=categorical_columns, drop_first=True)

# Apply the same preprocessing to the user input
user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

# Make predictions on the user input
prediction = model.predict(user_input)

# Display the prediction immediately
st.subheader("Prediction:")

# Display prediction using different styles based on the result
if prediction[0] == 1:
    st.success("The model predicts Conversion for the given input.")
else:
    st.info("The model predicts No Conversion for the given input.")

# Display additional information
st.subheader("Additional Information:")
st.write(f"Campaign Name: {campaign_name}")
st.write(f"Impressions: {impressions}")
st.write(f"Clicks: {clicks}")
st.write(f"Cost: ${cost:.2f}")
st.write(f"ROAS: {roas:.2f}")
st.write(f"Ad Creative Performance: {ad_creative_performance}")
st.write(f"Start Date: {start_date}")
st.write(f"End Date: {end_date}")

# Display prediction results based on inputs
st.subheader("Prediction Results based on User Inputs:")

# Display input values
st.write("**User Input Values:**")
st.write(user_input)

# Display prediction result
st.write("**Model Prediction:**")
if prediction[0] == 1:
    st.success("Conversion")
else:
    st.info("No Conversion")
