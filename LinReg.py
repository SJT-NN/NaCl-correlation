import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📊 Excel Correlation & Regression Visualizer")

# --- Upload Excel file ---
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # --- Select columns ---
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)

    # Optional: select column for y-error values
    yerr_col = st.selectbox(
        "Select Y-error column (optional)",
        ["None"] + cols
    )

    # --- Options ---
    through_origin = st.checkbox("Force regression through (0,0)")
    show_interval = st.checkbox("Show ±20% interval in green")
    interval_source = st.selectbox(
        "Interval source",
        ["Regression line", "y = x identity line"]
    )

    if x_col and y_col:
        # Prepare data
        X = df[[x_col]].dropna()
        y = df[y_col].dropna()

        # Align indices in case of NaNs
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # If y-error column selected, align it too
        yerr = None
        if yerr_col != "None":
            yerr_data = df[yerr_col].dropna()
            common_idx = common_idx.intersection(yerr_data.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            yerr = yerr_data.loc[common_idx].values

        # --- Regression model ---
        if through_origin:
            model = LinearRegression(fit_intercept=False)
        else:
            model = LinearRegression()

        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # --- Plot ---
        fig, ax = plt.subplots()

        if yerr is not None:
            ax.errorbar(X[x_col], y, yerr=yerr, fmt='o', alpha=0.7, label="Data points with error")
        else:
            ax.scatter(X, y, label="Data points", alpha=0.7)

        ax.plot(X, y_pred, color="red", linewidth=2, label="Regression line")

        # Always show y = x for reference
        x_range = np.linspace(X.min()[0], X.max()[0], 500)
        ax.plot(x_range, x_range, color="gray", linestyle="--", label="y = x")

        # --- Optional ±20% interval ---
        if show_interval:
            if interval_source == "Regression line":
                # Band from regression line
                y_fit = model.predict(x_range.reshape(-1, 1))
                y_plus = y_fit * 1.2
                y_minus = y_fit * 0.8
            else:
                # Band from identity line
                y_identity = x_range
                y_plus = y_identity * 1.2
                y_minus = y_identity * 0.8

            ax.fill_between(
                x_range, y_minus, y_plus,
                color="green", alpha=0.2,
                label=f"±20% from {interval_source.lower()}"
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
