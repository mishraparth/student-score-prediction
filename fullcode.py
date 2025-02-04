import pandas as pd

# Load the dataset (assuming 'india_housing_prices.csv' is your file)
df = pd.read_csv('StudentPerformanceFactors.csv')

# Get a summary of the DataFrame
print("DataFrame Information:")
print(df.info())

# Get descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# List all column names
print("\nColumn Names:")
print(df.columns)

# List data types of each column
print("\nData Types:")
print(df.dtypes)

# Display the first few rows of the DataFrame
print("\nFirst Few Rows:")
print(df.head())

# Display the last few rows of the DataFrame
print("\nLast Few Rows:")
print(df.tail())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Step 2: Preprocess the data
# Remove unnecessary columns
df = df.drop(['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home'], axis=1)

# Define features and target variable
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# Define categorical and numerical features
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                        'Motivation_Level', 'Internet_Access', 'Family_Income', 'School_Type', 
                        'Peer_Influence', 'Learning_Disabilities', 'Gender']
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                      'Tutoring_Sessions', 'Physical_Activity']

# Create a ColumnTransformer to handle categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that includes preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model_pipeline.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred_train = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f'Train MAE: {train_mae}')
print(f'Test MAE: {test_mae}')
print(f'Train R^2: {train_r2}')
print(f'Test R^2: {test_r2}')

# Step 6: Make Predictions
def predict_exam_score(input_data):
    input_df = pd.DataFrame([input_data])
    predicted_score = model_pipeline.predict(input_df)
    return predicted_score[0]

# Example usage:
input_data = {
    'Hours_Studied': 20,
    'Attendance': 80,
    'Parental_Involvement': 'Medium',
    'Access_to_Resources': 'High',
    'Extracurricular_Activities': 'Yes',
    'Sleep_Hours': 7,
    'Previous_Scores': 75,
    'Motivation_Level': 'Medium',
    'Internet_Access': 'Yes',
    'Tutoring_Sessions': 1,
    'Family_Income': 'Medium',
    'School_Type': 'Public',
    'Peer_Influence': 'Positive',
    'Physical_Activity': 3,
    'Learning_Disabilities': 'No',
    'Gender': 'Male'
}

predicted_score = predict_exam_score(input_data)
print(f'Predicted Exam Score: {predicted_score}')


import pickle
# Save the model pipeline
with open('model2.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

print("Model pipeline has been saved.")
