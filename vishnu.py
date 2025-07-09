import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_pipeline(numeric_features, categorical_features):
    """Create a pipeline for preprocessing."""
    # Numeric features pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

def transform_and_split(df, target_column, numeric_features, categorical_features, test_size=0.2, random_state=42):
    """Split data, fit and transform using preprocessing pipeline, and return results."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = preprocess_pipeline(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def load_pipeline_data(filepath, target_column, numeric_features, categorical_features):
    """Full pipeline: load data, preprocess, transform and split."""
    df = load_data(filepath)
    return transform_and_split(df, target_column, numeric_features, categorical_features)

# Example usage:
if __name__ == "__main__":
    # Update these according to your dataset
    filepath = "your_data.csv"
    target_column = "target"
    numeric_features = ["num_feature1", "num_feature2"]
    categorical_features = ["cat_feature1", "cat_feature2"]

    X_train, X_test, y_train, y_test, preprocessor = load_pipeline_data(
        filepath, target_column, numeric_features, categorical_features
    )

    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
