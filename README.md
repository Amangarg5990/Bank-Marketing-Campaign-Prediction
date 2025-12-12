# Bank Marketing Predictor

This project is a machine learning application designed to predict whether a client will subscribe to a term deposit based on their profile and previous campaign interactions. It utilizes the [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and presents the results through an interactive Streamlit dashboard.

## Features

-   **Interactive Dashboard**: A user-friendly web interface built with Streamlit.
-   **Model Comparison**: Evaluate and compare the performance of multiple machine learning models:
    -   K-Nearest Neighbors (KNN)
    -   Naive Bayes
    -   Logistic Regression
    -   Decision Tree
-   **Visualizations**: insightful charts including ROC curves, Feature Importance, and Confusion Matrices.
-   **Custom Prediction**: Input custom client data to get real-time predictions on term deposit subscription likelihood.

## Project Structure

-   `app.py`: The main entry point for the Streamlit web application.
-   `bank_marketing_ml.py`: Contains the core machine learning logic (data loading, preprocessing, training, and prediction functions).
-   `Bank_Marketing.csv`: The dataset used for training and testing the models.
-   `requirements.txt`: List of Python dependencies required to run the project.

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  The application will open in your default web browser (usually at `http://localhost:8501`).

3.  **Navigate the App**:
    -   **Sidebar**: Adjust user input features for custom predictions.
    -   **Prediction Tab**: Choose a model and see the prediction result for the sidebar inputs.
    -   **Model Evaluation Tab**: deeply analyze specific model performance with metrics and plots.
    -   **Conclusion Tab**: Compare all models side-by-side.
    -   **Code Tab**: View the underlying source code.

## Technologies Used

-   **Python**: Primary programming language.
-   **Streamlit**: Web application framework.
-   **Scikit-learn**: Machine learning library.
-   **Pandas**: Data manipulation and analysis.
-   **Matplotlib & Seaborn**: Data visualization.
