import streamlit as st
import pandas as pd
import numpy as np
import bank_marketing_ml as ml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

st.title("Bank Marketing Campaign Prediction")
st.markdown("""
This application predicts whether a client will subscribe to a term deposit based on their profile and campaign interactions.
""")

# Load Data (Cached)
@st.cache_data
def load_data():
    filepath = 'd:\\ml_ca_2\\Bank_Marketing.csv'
    return ml.load_and_preprocess_data(filepath)

# Load resources
try:
    df, X, y, preprocessor, le, num_cols, cat_cols = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Train Models (Cached)
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models, results = ml.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    return models, results

with st.spinner("Training models... (this may take a moment)"):
    trained_models, evaluation_results = train_models(X, y)

# Sidebar - User Input
# Sidebar - User Input

st.sidebar.header("User Input Features")

def user_input_features():
    input_data = {}
    
    # 1. Categorical Features
    for col in cat_cols:
        unique_vals = df[col].unique()
        # format_func to display options in Title Case (e.g. "admin." -> "Admin.")
        # Label is also Title Cased (e.g. "job" -> "Job")
        input_data[col] = st.sidebar.selectbox(f"{col.title()}", unique_vals, format_func=lambda x: str(x).title())
        
    # 2. Numerical Features
    for col in num_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        
        # Display label in Title Case
        label = col.title()
        
        if max_val - min_val > 100:
             input_data[col] = st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=mean_val)
        else:
             input_data[col] = st.sidebar.slider(label, min_value=min_val, max_value=max_val, value=mean_val)
             
    return input_data

input_features = user_input_features()

# Tabs
# Custom CSS to increase tab size
st.markdown("""
<style>
    div[data-baseweb="tab-list"] button {
        font_size: 24px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Evaluation", "Conclusion", "Code"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.header("Make a Prediction")
    st.write("Select a model and click probability to predict based on sidebar inputs.")
    
    model_choice_pred = st.selectbox("Choose Model for Prediction", list(trained_models.keys()), key="model_choice_pred")
    
    if st.button("Predict", key="predict_btn"):
        try:
            prediction = ml.predict_custom(
                input_features, 
                preprocessor, 
                trained_models[model_choice_pred], 
                le, 
                num_cols, 
                cat_cols
            )
            
            st.markdown(f"### Prediction Result: **{prediction.upper()}**")
            
            if prediction == 'yes':
                st.success("The client is likely to subscribe to a term deposit.")
            else:
                st.warning("The client is NOT likely to subscribe to a term deposit.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# --- TAB 2: MODEL EVALUATION ---
with tab2:
    st.header("Model Evaluation")
    
    model_name = st.selectbox("Choose Learning Model to Evaluate", list(trained_models.keys()), key="model_choice_eval")
    
    if model_name in evaluation_results:
        metrics = evaluation_results[model_name]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col3.metric("Recall", f"{metrics['Recall']:.4f}")
        col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        
        st.markdown("### Visualizations")
        
        # 1. ROC Curve
        try:
            from sklearn.metrics import roc_curve, auc
            model = trained_models[model_name]
            
            if hasattr(model, "predict_proba"):
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if model_name == 'Naive Bayes' and hasattr(X_test, "toarray"):
                    y_scores = model.predict_proba(X_test.toarray())[:, 1]
                else:
                    y_scores = model.predict_proba(X_test)[:, 1]
                    
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                
                # Styling for transparency
                fig_roc.patch.set_alpha(0.0)
                ax_roc.patch.set_alpha(0.0)
                ax_roc.xaxis.label.set_color('white')
                ax_roc.yaxis.label.set_color('white')
                ax_roc.title.set_color('white')
                ax_roc.tick_params(axis='x', colors='white')
                ax_roc.tick_params(axis='y', colors='white')
                for spine in ax_roc.spines.values():
                    spine.set_edgecolor('white')
                
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='white', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve: {model_name}')
                
                legend = ax_roc.legend(loc="lower right")
                legend.get_frame().set_alpha(0.0)
                for text in legend.get_texts():
                    text.set_color("white")
                    
                st.pyplot(fig_roc)
            else:
                st.info("ROC Curve not applicable (predict_proba not available).")
        except Exception as e:
            st.info(f"ROC Curve not available for this model: {e}")

        # 2. Feature Importance
        if model_name == 'Decision Tree':
            st.markdown("#### Feature Importance")
            importances = trained_models[model_name].feature_importances_
            try:
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                
                # Styling
                # Styling
                fig_imp.patch.set_alpha(0.0)
                ax_imp.patch.set_alpha(0.0)
                ax_imp.xaxis.label.set_color('white')
                ax_imp.yaxis.label.set_color('white')
                ax_imp.title.set_color('white')
                ax_imp.tick_params(axis='x', colors='white')
                ax_imp.tick_params(axis='y', colors='white')
                for spine in ax_imp.spines.values():
                    spine.set_edgecolor('white')
                
                sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', ax=ax_imp, palette='viridis', legend=False)
                ax_imp.set_title("Top 10 Feature Importances")
                st.pyplot(fig_imp)
            except Exception as e:
                st.warning(f"Could not plot feature importance: {e}")
                
        elif model_name == 'Logistic Regression':
            st.markdown("#### Feature Importance (Coefficients)")
            try:
                 model = trained_models[model_name]
                 if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = preprocessor.get_feature_names_out()
                    coefs = model.coef_[0]
                    
                    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
                    feat_imp_df['Abs_Val'] = feat_imp_df['Coefficient'].abs()
                    feat_imp_df = feat_imp_df.sort_values(by='Abs_Val', ascending=False).head(10)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    
                    # Styling
                    # Styling
                    fig_imp.patch.set_alpha(0.0)
                    ax_imp.patch.set_alpha(0.0)
                    ax_imp.xaxis.label.set_color('white')
                    ax_imp.yaxis.label.set_color('white')
                    ax_imp.title.set_color('white')
                    ax_imp.tick_params(axis='x', colors='white')
                    ax_imp.tick_params(axis='y', colors='white')
                    for spine in ax_imp.spines.values():
                        spine.set_edgecolor('white')
                    
                    sns.barplot(data=feat_imp_df, x='Coefficient', y='Feature', hue='Feature', ax=ax_imp, palette='coolwarm', legend=False)
                    ax_imp.set_title("Top 10 Feature Coefficients (Impact)")
                    st.pyplot(fig_imp)
            except Exception as e:
                pass

        st.markdown("#### Confusion Matrix")
        cm = metrics['Confusion Matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        
        # Styling
        # Styling
        fig_cm.patch.set_alpha(0.0)
        ax_cm.patch.set_alpha(0.0)
        ax_cm.xaxis.label.set_color('white')
        ax_cm.yaxis.label.set_color('white')
        ax_cm.title.set_color('white')
        ax_cm.tick_params(axis='x', colors='white')
        ax_cm.tick_params(axis='y', colors='white')
        for spine in ax_cm.spines.values():
            spine.set_edgecolor('white')
        
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='YlGnBu', ax=ax_cm, linewidths=.5, cbar=True)
        ax_cm.set_xlabel('Predicted Label', fontsize=12)
        ax_cm.set_ylabel('True Label', fontsize=12)
        ax_cm.set_title(f'Confusion Matrix: {model_name}', fontsize=14)
        st.pyplot(fig_cm)

# --- TAB 3: CONCLUSION ---
with tab3:
    st.header("Conclusion & Model Comparison")
    st.write("Comparison of performance metrics across all trained models.")
    
    # Prepare data for comparison
    comparison_data = []
    for m_name, metrics in evaluation_results.items():
        comparison_data.append({
            "Model": m_name,
            "Accuracy": metrics['Accuracy'],
            "Precision": metrics['Precision'],
            "Recall": metrics['Recall'],
            "F1 Score": metrics['F1 Score']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Reshape for grouped bar chart
    comparison_melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Plot using Seaborn/Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Styling for transparency
    # Styling for transparency
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    sns.barplot(data=comparison_melted, x="Model", y="Score", hue="Metric", ax=ax, palette="viridis")
    plt.ylim(0, 1.0)
    plt.title("Model Performance Metrics Comparison")
    
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_alpha(0.0)
    for text in legend.get_texts():
        text.set_color("white")
        
    st.pyplot(fig)
    
    st.markdown("### Findings")
    st.markdown("""
    - **Logistic Regression and KNN** tend to have high accuracy, but it's important to look at Recall for this imbalanced dataset.
    - **Naive Bayes** often has better Recall but lower Precision.
    - **Decision Tree** provides a balance and interpretability via feature importance.
    
    For a marketing campaign, **Recall** is often critical (we don't want to miss potential subscribers), so check which model maximizes Recall while maintaining acceptable Precision.
    """)

# --- TAB 4: CODE ---
with tab4:
    st.header("Source Code")
    
    st.subheader("1. Streamlit Application (app.py)")
    try:
        with open(__file__, "r") as f:
            st.code(f.read(), language="python")
    except Exception as e:
        st.error(f"Could not load app.py: {e}")
        
    st.subheader("2. ML Logic (bank_marketing_ml.py)")
    try:
        # Assuming the file is in the same directory
        import os
        ml_file_path = os.path.join(os.path.dirname(__file__), "bank_marketing_ml.py")
        with open(ml_file_path, "r") as f:
            st.code(f.read(), language="python")
    except Exception as e:
        st.error(f"Could not load bank_marketing_ml.py: {e}")

# --- FOOTER ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    .footer-container {
        width: 100%;
        text-align: center;
        padding: 20px 0;
        margin-top: 50px;
        border-top: 1px solid #333;
        font-family: 'Arial', sans-serif;
    }
    .footer-text {
        font-size: 16px;
        margin-bottom: 10px;
        color: #888;
    }
    .footer-icons {
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .footer-icon {
        font-size: 24px;
        color: #888;
        transition: color 0.3s ease;
    }
    .footer-icon:hover {
        color: #4CAF50; /* Green hover effect */
    }
</style>

<div class="footer-container">
    <div class="footer-text">
        Made by <strong>Aman Garg</strong>
    </div>
    <div class="footer-icons">
        <a href="https://www.linkedin.com/in/amangarg09/" target="_blank" class="footer-icon">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/Amangarg5990" target="_blank" class="footer-icon">
            <i class="fab fa-github"></i>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)