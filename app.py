import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- Configuration and File Paths ---
MODEL_FILE = 'best_lo_model (1).pkl'
DATA_FILE = 'insurance.csv'

# --- Data and Model Loading ---

@st.cache_resource(show_spinner="Loading pre-trained prediction model...")
def load_prediction_model(path):
    """Loads the pickled scikit-learn model."""
    try:
        # Use 'rb' for reading binary files
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure the file is correctly available.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(show_spinner="Loading insurance data...")
def load_dataset(path):
    """Loads the insurance CSV dataset."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {path}. Please ensure the file is correctly available.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load resources
insurance_model = load_prediction_model(MODEL_FILE)
insurance_df = load_dataset(DATA_FILE)

# --- Streamlit Application Layout ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Insurance Charges Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏥 Health Insurance Charges Predictor")
    st.markdown("""
        Estimate your annual medical insurance costs based on key health factors.
        The prediction uses a Linear Regression model trained on the provided dataset.
    """)

    if insurance_model is None or insurance_df.empty:
        st.error("Application requires both the model and data files to function.")
        return

    # --- Sidebar for User Input ---
    st.sidebar.header("👤 Your Profile")
    st.sidebar.markdown("---")

    # 1. Age
    age = st.sidebar.slider(
        "Age", 
        min_value=18, 
        max_value=64, 
        value=30, 
        help="Your age in years (min 18, max 64 in the training data)."
    )

    # 2. BMI
    min_bmi = float(insurance_df['bmi'].min())
    max_bmi = float(insurance_df['bmi'].max())
    default_bmi = float(insurance_df['bmi'].mean())
    bmi = st.sidebar.slider(
        "BMI (Body Mass Index)", 
        min_value=min_bmi, 
        max_value=max_bmi, 
        value=default_bmi, 
        step=0.1,
        format="%.1f",
        help=f"Your BMI. (Dataset range: {min_bmi:.1f} to {max_bmi:.1f})"
    )

    # 3. Smoker
    smoker_option = st.sidebar.selectbox(
        "Smoker Status", 
        ('No', 'Yes'), 
        help="Do you currently smoke?"
    )
    # Convert 'Yes'/'No' to the model's expected input (smoker_yes: 1 or 0)
    smoker_yes = 1 if smoker_option == 'Yes' else 0

    # Information about unused features
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Note")
    st.sidebar.info("This specific model only uses **Age**, **BMI**, and **Smoker Status** for prediction. Other factors like number of children, sex, and region are not used.")


    # --- Prediction Button and Logic ---
    st.sidebar.markdown("---")
    if st.sidebar.button("💸 Calculate Estimated Charges", type="primary"):
        # Create a DataFrame for prediction, ensuring correct column order
        # Expected features by the loaded model are: ['age', 'bmi', 'smoker_yes']
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'smoker_yes': [smoker_yes]
        })

        # Make prediction
        try:
            prediction = insurance_model.predict(input_data)[0]

            # Display Result
            st.header("💰 Estimated Annual Charges")
            st.markdown(f"""
                <div style="padding: 30px; border-radius: 12px; background-color: #00796b; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <p style="font-size: 1.2em; margin-bottom: 5px;">Your predicted cost is:</p>
                    <h1 style="font-size: 3em; margin: 0;">${prediction:,.2f}</h1>
                    <p style="font-size: 0.9em; margin-top: 5px;">(Based on a Linear Regression Model)</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Contextual Message
            if smoker_yes == 1:
                st.warning("🚨 Prediction is significantly higher due to **Smoker Status**.")
            elif bmi >= 30:
                st.info("💡 Note: A BMI of 30 or higher is typically classified as obese, which can increase charges.")
            else:
                st.success("✅ Prediction suggests lower charges for non-smokers.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # --- Main Content: Data and Model Insights ---
    st.markdown("---")
    st.header("📊 Data & Model Insights")
    
    if not insurance_df.empty:
        tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Charges Distribution", "Feature Impact"])
        
        with tab1:
            st.subheader("Raw Data Sample (`insurance.csv`)")
            st.dataframe(insurance_df.head(10))
            st.caption(f"The dataset contains {len(insurance_df)} customer records.")

        with tab2:
            st.subheader("Charges Distribution")
            
            # Simple bar chart of charges by smoker status
            smoker_charges = insurance_df.groupby('smoker')['charges'].mean().reset_index()
            smoker_charges.columns = ['Smoker', 'Average Charges']
            
            st.bar_chart(smoker_charges.set_index('Smoker'))
            st.caption("Average charges are drastically higher for smokers.")
            
            # Histogram for overall charges
            st.subheader("Charges Frequency")
            st.plotly_chart(
                pd.DataFrame({
                    "Charges ($)": insurance_df['charges']
                }).plot(kind='hist', bins=25, title='Distribution of Annual Charges'),
                use_container_width=True
            )

        with tab3:
            st.subheader("Model Feature Explanation")
            st.markdown("""
            The loaded model is a **Linear Regression** model.
            It uses the following three features to calculate the estimated cost:
            
            1.  **Age:** Directly proportional to cost.
            2.  **BMI:** Higher BMI leads to higher costs.
            3.  **Smoker:** The most significant factor, adding a large fixed amount to the base charge if 'Yes'.
            
            The model's coefficients represent the dollar amount added to the charge for a one-unit increase in that feature (e.g., the cost increase per year of age or per BMI point).
            """)

# Run the app
if __name__ == "__main__":
    main()
