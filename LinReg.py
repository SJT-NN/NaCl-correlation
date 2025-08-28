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
    # Get sheet names
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    # Select sheet
    sheet_name = st.selectbox("Select sheet", sheet_names)

    # Load chosen sheet
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    st.subheader(f"Preview of Data — Sheet: {sheet_name}")
    st.dataframe(df.head())

    # --- Column selectors ---
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)
    yerr_col = st.selectbox("Select Y-error column (optional)", ["None"] + cols)

    # --- Display options ---
    point_size = st.slider("Scatter point size", 10, 200, 50)
    plot_width = st.slider("Plot width (inches)", 4, 16, 8)
    plot_height = st.slider("Plot height (inches)", 4, 12, 6)

    # --- Analysis options ---
    through_origin = st.checkbox("Force regression through (0,0)")
    show_interval = st.checkbox("Show ±20% interval in green")
    interval_source = st.selectbox(
        "Interval source",
        ["Regression line", "y = x identity line"]
    )

    if x_col and y_col:
        # --- Clean data once ---
        required_cols = [x_col, y_col] if yerr_col == "None" else [x_col, y_col, yerr_col]
        df_valid = df[required_cols].dropna()

        X = df_valid[[x_col]]
        y = df_valid[y_col]
        yerr = df_valid[yerr_col].values if yerr_col != "None" else None

        # --- Fit regression ---
        model = LinearRegression(fit_intercept=not through_origin)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        slope_val = model.coef_[0]
        intercept_val = model.intercept_ if not through_origin else 0

        # --- Prepare X for smooth lines ---
        x_min, x_max = X[x_col].min(), X[x_col].max()
        x_range = np.linspace(x_min, x_max, 500)
        sorted_idx = np.argsort(X[x_col])

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))

        if yerr is not None:
            ax.errorbar(
                X[x_col], y,
                yerr=yerr,
                fmt='o',
                alpha=0.7,
                markersize=point_size / 10,
                label="Data points with error"
            )
        else:
            ax.scatter(
                X, y,
                s=point_size,
                alpha=0.7,
                label="Data points"
            )

        # Regression line
        ax.plot(
            X.iloc[sorted_idx], y_pred[sorted_idx],
            color="red", linewidth=2,
            label=f"Regression line (slope={slope_val:.4f}, intercept={intercept_val:.4f}, R²={r2:.4f})"
        )

        # y = x reference line
        ax.plot(x_range, x_range, color="gray", linestyle="--", label="y = x")

        # --- ±20% interval ---
        if show_interval:
            if interval_source == "Regression line":
                y_fit = model.predict(x_range.reshape(-1, 1))
                y_plus, y_minus = y_fit * 1.2, y_fit * 0.8
            else:
                y_plus, y_minus = x_range * 1.2, x_range * 0.8

            ax.fill_between(
                x_range, y_minus, y_plus,
                color="green", alpha=0.2,
                label=f"±20% from {interval_source.lower()}"
            )

        # Labels & legend
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        # --- Numeric results ---
        st.markdown(f"**Slope:** {slope_val:.4f}")
        st.markdown(f"**Intercept:** {intercept_val:.4f}" if not through_origin else "**Intercept:** forced to 0")
        st.markdown(f"**R² score:** {r2:.4f}")
