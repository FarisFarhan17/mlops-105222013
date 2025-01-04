import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
# Load Titanic dataset from local file
file_path = 'data/titanic.csv' 
data = pd.read_csv(file_path)

# Preprocess the dataset (e.g., filling missing values, encoding categorical variables)
data['Age'] = data['Age'].fillna(data['Age'].median())  
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) 

# Split the dataset (train/test split)
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run(run_name="Titanic_Dataset_Logging"):
    # Log the dataset as an artifact (the raw CSV file)
    mlflow.log_artifact(file_path)  
    
    # Log dataset statistics
    mlflow.log_param("num_features", X.shape[1])  
    mlflow.log_param("num_samples", X.shape[0])   

    # Log some basic statistics (e.g., mean and std of features) only for numeric columns
    for column in X.select_dtypes(include=['number']).columns:  
        mlflow.log_metric(f"{column}_mean", X[column].mean())
        mlflow.log_metric(f"{column}_std", X[column].std())

    # Log feature names for further model drift detection or analysis
    feature_names = X.columns.tolist()
    mlflow.log_dict({'feature_names': feature_names}, 'feature_names.json')

    print("Dataset logged successfully!")
    print(mlflow.get_tracking_uri())

