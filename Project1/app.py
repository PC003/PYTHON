import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import train_test_split

st.title("Upload Cleaned Dataset!")

file_upload=st.file_uploader("Upload file",type=["csv"])

st.title("What in tha data!!!")
if file_upload is not None:
    df = pd.read_csv(file_upload)
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
   
    tab1, tab2 ,tab3,tab4= st.tabs(["📄 Data Preview", "📊 Visualizations","Staticstics","Models"])
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    with tab2:
        chart_type = st.selectbox(
            "Select chart type",
            ["Bar Chart", "Line Chart", "Scatter Plot"]
        )

        if chart_type in ["Bar Chart", "Line Chart"]:
            x = st.selectbox("Select X axis", numeric_columns)
            y = st.selectbox("Select Y axis", numeric_columns)

        if chart_type == "Scatter Plot":
            x = st.selectbox("Select X axis", numeric_columns)
            y = st.selectbox("Select Y axis", numeric_columns)

        if st.button("Generate Chart"):
            if chart_type == "Bar Chart":
                st.bar_chart(df, x=x, y=y)

            elif chart_type == "Line Chart":
                st.line_chart(df, x=x, y=y)

            elif chart_type == "Scatter Plot":
                st.scatter_chart(df, x=x, y=y)    
    with tab3:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    with tab4:
        st.subheader("Regression Model")

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        non=df.select_dtypes(include="object").columns.to_list()
        # Select target
        predicted_column = st.selectbox(
            "Select Output Variable (Y)",
            numeric_columns
        )

        # Prepare data
        df_copy = df.copy()
        X = df_copy.drop(columns=[predicted_column]+non)
        y = df_copy[predicted_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"]
        )

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Lasso Regression":
            model = Lasso()
        elif model_type == "Ridge Regression":
            model = Ridge()
        else:
            model = ElasticNet()

        model.fit(X_train, y_train)

        st.markdown("### Enter Input Values")

        input_data = []

        
        for col in X.columns:
            val = st.number_input(
                f"{col}",
                value=float(X[col].mean())
            )
            input_data.append(val)

       
        input_df = pd.DataFrame([input_data], columns=X.columns)

        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.success(f"Predicted Value: {prediction[0]:.4f}")