import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Set MLflow tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Load dataset
file_path = 'data/titanic.csv'
data = pd.read_csv(file_path)

# Preprocess the dataset
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Encode categorical columns: Sex and Embarked
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

# Encode 'Embarked' using get_dummies (one-hot encoding)
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Split dataset into features and target
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and log models
def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=f"{model_name}_Model"):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc_score", auc_score)
        
        # Log the trained model
        mlflow.sklearn.log_model(model, model_name)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
train_and_log_model(logreg, "Logistic_Regression")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
train_and_log_model(rf, "Random_Forest")

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
train_and_log_model(nn, "Neural_Network")

print("All models have been trained and logged!")
