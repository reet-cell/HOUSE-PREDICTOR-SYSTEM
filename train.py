import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'Neighborhood']
target = 'SalePrice'

df = df[features + [target]]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
cat_features = ['Neighborhood']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

pickle.dump(pipeline, open("pipeline.pkl", "wb"))

print("Model saved!")
print("Training started...")

pipeline.fit(X_train, y_train)

print("Training completed!")

pickle.dump(pipeline, open("pipeline.pkl", "wb"))

print("Model saved successfully!")