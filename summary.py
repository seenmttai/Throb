import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
data_df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

# Set the target variable and drop irrelevant columns
X = data_df.drop(columns=["HeartDiseaseorAttack", "Education", "Income", "DiffWalk", "Fruits", "Veggies"])

# Define numeric and categorical columns
num_cols = ["BMI", "MentHlth", "PhysHlth", "Age"]
cat_cols = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
            "Diabetes", "PhysActivity", "HvyAlcoholConsump", "AnyHealthcare", 
            "NoDocbcCost", "GenHlth", "Sex"]

# Ensure the column lists match the available columns in X
num_cols = [col for col in num_cols if col in X.columns]
cat_cols = [col for col in cat_cols if col in X.columns]

# Create pipelines for preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine the pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Fit the preprocessor to the data
preprocessor.fit(X)

# Save the preprocessor to a file
joblib.dump(preprocessor, "preprocessor.pkl")

print("Preprocessor saved as preprocessor.pkl")
