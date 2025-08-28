import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📈 Excel Correlation & Linear Regression Tool")

# --- File upload ---
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # --- Column selection ---
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)

    # --- Regression options ---
    through_origin = st.checkbox("Force regression through (0,0)")
    show_interval = st.checkbox("Show ±20% interval in green")

    if x_col and y_col:
        # Prepare data
        X = df[[x_col]].dropna()
        y = df[y_col].dropna()

        # Align indices if missing values
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Model
        if through_origin:
            model = LinearRegression(fit_intercept=False)
        else:
            model = LinearRegression()

        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="Data points", alpha=0.7)
        ax.plot(X, y_pred, color="red", linewidth=2, label="Regression line")

        if show_interval:
            # Create smooth x range for band
            x_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
            y_fit = model.predict(x_range)

            # Define ±20% band
            y_plus = y_fit * 1.2
            y_minus = y_fit * 0.8

            # Fill between for the green zone
            ax.fill_between(
                x_range.flatten(), y_minus, y_plus,
                color='green', alpha=0.2,
                label="±20% range"
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        # --- Results ---
        st.markdown(f"**Slope:** {model.coef_[0]:.4f}")
        if not through_origin:
            st.markdown(f"**Intercept:** {model.intercept_:.4f}")
        else:
            st.markdown("**Intercept:** forced to 0")
        st.markdown(f"**R² score:** {r2:.4f}")
