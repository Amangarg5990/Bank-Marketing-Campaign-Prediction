import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Preprocessing for numerical data: Standard Scaling
    # Preprocessing for categorical data: One-Hot Encoding
    # We use ColumnTransformer to apply different preprocessing to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Preprocess features
    print("Preprocessing features...")
    X_processed = preprocessor.fit_transform(X)
    
    # Encode target variable (no/yes -> 0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return df, X_processed, y_encoded, preprocessor, le, numerical_cols, categorical_cols

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(), # GaussianNB typically requires dense input, we might need to densify
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # GaussianNB requires dense matrix
        if name == 'Naive Bayes' and hasattr(X_train, "toarray"):
             model.fit(X_train.toarray(), y_train)
             y_pred = model.predict(X_test.toarray())
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        trained_models[name] = model
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': conf_matrix
        }
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
    return trained_models, results

def predict_custom(input_data, preprocessor, model, le, numerical_cols, categorical_cols):
    """
    Predicts outcome for a custom input using the trained model and preprocessor.
    input_data: dict containing feature values
    """
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_processed = preprocessor.transform(input_df)
    
    # Predict
    # Check if model requires dense input (GaussianNB) - though we'll likely use DecisionTree or LR for best interpretable results or generalized
    if isinstance(model, GaussianNB) and hasattr(input_processed, "toarray"):
         prediction_encoded = model.predict(input_processed.toarray())
    else:
        prediction_encoded = model.predict(input_processed)
        
    prediction = le.inverse_transform(prediction_encoded)
    return prediction[0]

def main():
    filepath = os.path.join(os.path.dirname(__file__), 'Bank_Marketing.csv')
    
    # 1. Load and Preprocess
    _, X, y, preprocessor, le, num_cols, cat_cols = load_and_preprocess_data(filepath)
    
    # 2. Split Data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train and Evaluate
    trained_models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 4. Custom Prediction Loop
    print("\n--- Custom Prediction Demo ---")
    
    # Example custom input (taking the first row of actual data as an example, but modifying slightly)
    # age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome
    # 58,management,married,tertiary,no,2143,yes,no,unknown,5,may,261,1,-1,0,unknown
    
    custom_input = {
        'age': 58,
        'job': 'management',
        'marital': 'married',
        'education': 'tertiary',
        'default': 'no',
        'balance': 2143,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'unknown',
        'day': 5,
        'month': 'may',
        'duration': 261,
        'campaign': 1,
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown'
    }
    
    print(f"Input features: {custom_input}")
    
    # Using Decision Tree for prediction demo as it's often user-friendly
    model_name = 'Decision Tree'
    prediction = predict_custom(custom_input, preprocessor, trained_models[model_name], le, num_cols, cat_cols)
    print(f"Prediction using {model_name}: {prediction}")

    # Allow user to input custom values via terminal if run interactively (commented out for automation flow, but structure is here)
    # while True:
    #    ...

if __name__ == "__main__":
    main()
