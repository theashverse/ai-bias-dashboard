import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import altair as alt
from fairlearn.metrics import demographic_parity_difference
from fairlearn.preprocessing import CorrelationRemover
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Initialize Session State ---
if 'target_feature' not in st.session_state:
    st.session_state['target_feature'] = None
if 'favorable_outcome' not in st.session_state:
    st.session_state['favorable_outcome'] = None
if 'sensitive_feature' not in st.session_state:
    st.session_state['sensitive_feature'] = None

# 1. Set the page configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Unbiased AI Dashboard", page_icon="🐦‍🔥", layout="wide")

# 2. Dashboard Header
st.title("🐦‍🔥 AI Fairness & Bias Auditing Dashboard")
st.markdown("Upload any dataset to audit for hidden biases, evaluate fairness metrics, and mathematically reweigh your data before training AI models.")

st.divider() # Adds a visual horizontal line

# 3. Section 1: Data Ingestion
st.header("1. Data Ingestion")
# UPGRADE: Added xlsx and json to the allowed types
uploaded_file = st.file_uploader("Upload your historical dataset (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])

# 4. Handle the uploaded file dynamically
if uploaded_file is not None:
    # UPGRADE: A try/except block prevents the app from crashing if the file is corrupted
    try:
        # Check the file extension to use the right Pandas reader
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext == 'xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
            
        st.success(f"Successfully loaded: {uploaded_file.name}")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head()) 
        
        st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")
        st.session_state['raw_data'] = df
        
    except Exception as e:
        # UPGRADE: Graceful error handling
        st.error(f"⚠️ Could not read the file. Please ensure it is a valid tabular format. Error details: {e}")
else:
    st.info("Please upload a CSV, Excel, or JSON file to begin the audit.")

st.divider()

# 5. Section 2: AI Bias Audit (Gemini Integration)
st.header("2. AI Proxy Bias Detection")
st.markdown("Use Generative AI to analyze your dataset's columns for direct and hidden proxy biases.")


# UPGRADE 1: Ask the user for the dataset context
dataset_context = st.text_input("What is the purpose of this dataset? (e.g., Resume screening, Loan approvals, Healthcare diagnosis)", placeholder="Briefly describe what this AI model will decide...")

# NEW UPGRADE: Dropdown for model selection with an "Other" fallback
model_options = [
    "Random Forest", 
    "Logistic Regression", 
    "XGBoost", 
    "Support Vector Machine (SVM)", 
    "Neural Network (Deep Learning)",
    "Large Language Model (e.g., GPT, Gemini)",
    "Other"
]
selected_model = st.selectbox("What ML Model or Algorithm are you planning to use?", options=model_options)

# If they select "Other", show the text box. Otherwise, use their selection.
if selected_model == "Other":
    model_type = st.text_input("Please specify your custom model architecture:")
else:
    model_type = selected_model

# Create a secure text input for the API Key
api_key = st.text_input("Enter your Google Gemini API Key:", type="password", help="Get this from Google AI Studio")

# Only show the "Run Audit" button if a file is uploaded AND an API key is entered
if api_key and 'raw_data' in st.session_state:
    if st.button("Run AI Bias Audit", type="primary"):
        
        with st.spinner("Gemini is analyzing your data columns..."):
            try:
                genai.configure(api_key=api_key)
                # Using the modern 2.5 Flash model
                model = genai.GenerativeModel('gemini-2.5-flash') 
                
                df = st.session_state['raw_data']
                columns_list = df.columns.tolist()
                
                # UPGRADE: Smart Auto-Detect for missing context
                if dataset_context:
                    context_instruction = f"Dataset Context: {dataset_context}"
                else:
                    context_instruction = "Dataset Context: Not provided. Please analyze the column names, infer the most likely industry and purpose of this dataset (e.g., Healthcare, Finance, HR), state your assumed context at the top of your response, and base your risk analysis on that assumption."

                prompt = f"""
                Act as a Data Ethics and Legal Compliance Expert. 
                I am going to give you a list of column headers from a dataset, along with the project context and ML model type.
                
                {context_instruction}
                Model Architecture: {model_type if model_type else 'Not provided.'}
                
                Task 1 - Data Inspection: Identify any attributes in the columns that could lead to direct discrimination or hidden proxy discrimination based on the context (or your assumed context).
                Task 2 - Model Inspection: Briefly explain any known algorithmic biases or fairness risks associated with using the specified Model Architecture.
                
                Format your response as a clean, highly readable report divided into "Context & Assumptions", "Data Risks", and "Model Risks". 
                
                Dataset Columns to analyze: {columns_list}
                """
                
                response = model.generate_content(prompt)
                st.success("AI Audit Complete!")
                
                with st.container(border=True):
                    st.markdown(response.text)
                
            except Exception as e:
                st.error(f"⚠️ Google Gemini API Error. Please check your API key and try again. Details: {e}")

elif not api_key:
    st.warning("⚠️ Please enter your API key to unlock the AI Audit feature.")

st.divider()
# ---------------------------------------------------------
# Section 3: The Hybrid Bias Discovery Engine
# ---------------------------------------------------------

# SAFETY CHECK: Only show Section 3 if data has been uploaded
if 'raw_data' in st.session_state:
    df = st.session_state['raw_data']
    
    st.header("3. Automated Bias Discovery Engine")
    st.markdown("Instead of guessing where the bias is hiding, let our hybrid AI scanner find it for you.")

    # Step 3.1: Define the Goal
    st.markdown("#### Step A: Define the Business Goal")
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter out demographic traits from the outcome list to prevent circular logic
        prohibited_outcomes = ['race', 'sex', 'gender', 'age', 'marital.status', 'native.country', 'religion']
        outcome_options = [col for col in df.columns if not any(word in col.lower() for word in prohibited_outcomes)]

        target_feature = st.selectbox(
            "Select Outcome Feature (The Business Goal):", 
            options=outcome_options,
            key="target_sel"
        )
    with col2:
        favorable_options = df[target_feature].dropna().unique()
        favorable_outcome = st.selectbox("Select Favorable Outcome (The 'Win'):", options=favorable_options, key="fav_sel")

    # Step 3.2: The Auto-Scanner
    st.markdown("#### Step B: Run the Threat Scanner")
    
    # Check for API Key before allowing the scan
    if not api_key:
        st.info("💡 Enter your Gemini API key in Section 2 to enable the scanner.")
    else:
        if st.button("🔍 Scan Dataset for Hidden Bias", type="primary"):
            with st.spinner("Analyzing for bias..."):
                try:
                    # 1. IDENTIFY CATEGORICAL COLUMNS
                    categorical_cols = [col for col in df.columns if col != target_feature and (df[col].dtype == 'object' or str(df[col].dtype) == 'category' or df[col].nunique() < 10)]
                    
                    # 2. GEMINI ETHICS SCAN
                    genai.configure(api_key=api_key)
                    scan_model = genai.GenerativeModel('gemini-2.5-flash') # Corrected model name

                    prompt = f"Identify sensitive demographic traits (gender, race, age, etc.) from this list: {categorical_cols}. Return ONLY a comma-separated list of the column names."
                    response = scan_model.generate_content(prompt)
                    
                    gemini_suggestions = [x.strip() for x in response.text.replace('`', '').split(',')]
                    sensitive_cols = [col for col in gemini_suggestions if col in df.columns]
                    
                    if not sensitive_cols:
                        sensitive_cols = categorical_cols # Fallback
                        
                    # 3. FAIRLEARN MATH
                    threat_report = []
                    clean_df_scan = df.dropna(subset=[target_feature] + sensitive_cols)
                    y_scan = (clean_df_scan[target_feature] == favorable_outcome).astype(int)
                    
                    for sens_col in sensitive_cols:
                        dp_diff = demographic_parity_difference(y_scan, y_scan, sensitive_features=clean_df_scan[sens_col])
                        threat_report.append({"Sensitive Feature": sens_col, "Demographic Parity Difference": dp_diff})
                    
                    # Store results in Session State
                    st.session_state['threat_report'] = pd.DataFrame(threat_report).sort_values(by="Demographic Parity Difference", ascending=False)
                    st.session_state['target_feature'] = target_feature
                    st.session_state['favorable_outcome'] = favorable_outcome
                    
                except Exception as e:
                    st.error(f"Scanner error: {e}")

    # Step 3.3: Findings & Visuals
    if 'threat_report' in st.session_state:
        st.markdown("### 🚨 Threat Report Findings")
        report_df = st.session_state['threat_report']
        
        st.dataframe(
            report_df.style.format({"Demographic Parity Difference": "{:.3f}"})
            .map(lambda x: 'background-color: #ffcccc; color: black' if x > 0.1 else 'background-color: #ccffcc; color: black', subset=['Demographic Parity Difference']),
            use_container_width=True
        )
        
        st.markdown("#### Step C: Select Target for Mitigation")
        selected_threat = st.selectbox(
            "Which sensitive feature would you like to neutralize in Step 4?", 
            options=report_df['Sensitive Feature'].tolist()
        )
        st.session_state['sensitive_feature'] = selected_threat
        
        # Visual Chart
        st.markdown(f"**📊 Visualizing Success Rates by {selected_threat}**")
        chart_data = df.dropna(subset=[selected_threat, st.session_state['target_feature']])
        success_rates = chart_data.groupby(selected_threat).apply(
            lambda x: (x[st.session_state['target_feature']] == st.session_state['favorable_outcome']).mean() * 100
        ).reset_index(name='Success Rate (%)')
        
        # Rename column for Altair safety
        success_rates.rename(columns={selected_threat: 'Group'}, inplace=True)
        
        chart = alt.Chart(success_rates).mark_bar(color='#4A90E2').encode(
            x=alt.X('Group:N', title=selected_threat, sort='-y'),
            y=alt.Y('Success Rate (%):Q', title='Success Rate (%)'),
            tooltip=['Group', 'Success Rate (%)']
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
# ---------------------------------------------------------
# Section 4: Mitigation Logic & Result Display
# ---------------------------------------------------------
st.divider()
st.header("4. Bias Mitigation & Accuracy Check")

if 'target_feature' not in st.session_state or st.session_state['target_feature'] is None:
    st.warning("⚠️ Complete Section 3 to unlock mitigation.")
else:
    target_feature = st.session_state['target_feature']
    favorable_outcome = st.session_state['favorable_outcome']
    sensitive_feature = st.session_state['sensitive_feature']

    if st.button("🛠️ Mitigate Bias", type="primary"):
        try:
            # 1. Prepare data
            df_clean = st.session_state['raw_data'].copy().dropna(subset=[sensitive_feature, target_feature])
            y_target = (df_clean[target_feature] == favorable_outcome).astype(int)
            
            # 2. Encode categorical columns
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object' or str(df_clean[col].dtype) == 'category':
                    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
            
            # 3. Separate Target and Features
            X = df_clean.drop(columns=[target_feature])
            
            # 4. Neutralize the data
            cr = CorrelationRemover(sensitive_feature_ids=[sensitive_feature])
            X_neutralized = cr.fit_transform(X)
            
            # 5. FIX THE SHAPE ERROR: Create column list WITHOUT the sensitive feature
            # CorrelationRemover drops the sensitive column from the array
            remaining_columns = [c for c in X.columns if c != sensitive_feature]
            
            # 6. Create the "Fair" Dataframe for display
            df_fair = pd.DataFrame(X_neutralized, columns=remaining_columns)
            df_fair[target_feature] = y_target.values # Add the decision back in
            
            # --- THE REVEAL ---
            st.success(f"✅ Bias successfully neutralized relative to '{sensitive_feature}'!")
            
            st.subheader("✨ Unbiased Dataset Preview")
            st.markdown("The values below are mathematically transformed to ensure zero correlation with the sensitive trait.")
            st.dataframe(df_fair.head(10), use_container_width=True)

            # 7. Proof of Utility (Accuracy Comparison)
            st.divider()
            st.subheader("🧪 Accuracy vs. Fairness Tradeoff")
            
            # Train on Original (Drop target, keep sensitive for a fair baseline)
            X_train_o, X_test_o, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
            model_old = RandomForestClassifier(random_state=42).fit(X_train_o, y_train)
            acc_old = accuracy_score(y_test, model_old.predict(X_test_o))
            
            # Train on Neutralized
            X_train_n, X_test_n, _, _ = train_test_split(X_neutralized, y_target, test_size=0.2, random_state=42)
            model_new = RandomForestClassifier(random_state=42).fit(X_train_n, y_train)
            acc_new = accuracy_score(y_test, model_new.predict(X_test_n))
            
            col_acc1, col_acc2 = st.columns(2)
            col_acc1.metric("Original Accuracy", f"{acc_old*100:.1f}%")
            col_acc2.metric("Fair Accuracy", f"{acc_new*100:.1f}%", delta=f"{(acc_new-acc_old)*100:.1f}%")

            # 8. Final Download
            st.download_button(
                label="⬇️ Download Neutralized CSV", 
                data=df_fair.to_csv(index=False).encode('utf-8'), 
                file_name="neutralized_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Mitigation error: {e}")

# This final check ensures Section 3 and 4 only appear if a file is uploaded
if 'raw_data' not in st.session_state:
    st.info("Waiting for dataset upload in Section 1...")