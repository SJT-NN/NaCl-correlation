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
    # Get all sheet names first
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    # If multiple sheets, choose one
    sheet_name = st.selectbox("Select sheet", sheet_names)

    # Load chosen sheet
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    st.subheader(f"Preview of Data — Sheet: {sheet_name}")
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

    # --- Display options ---
    point_size = st.slider("Scatter point size", min_value=10, max_value=200, value=50)
    plot_width = st.slider("Plot width (inches)", min_value=4, max_value=16, value=8)
    plot_height = st.slider("Plot height (inches)", min_value=4, max_value=12, value=6)

    # --- Analysis options ---
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

        slope_val = model.coef_[0]
        intercept_val = model.intercept_ if not through_origin else 0

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
                label="Data points",
                alpha=0.7
            )

        # Regression line with slope/intercept/R² in legend
        ax.plot(
            X, y_pred,
            color="red", linewidth=2,
            label=f"Regression line (slope={slope_val:.4f}, intercept={intercept_val:.4f}, R²={r2:.4f})"
        )

        # Always show y = x for reference
        x_range = np.linspace(X.min()[0], X.max()[0], 500)
        ax.plot(x_range, x_range, color="gray", linestyle="--", label="y = x")

        # --- Optional ±20% interval ---
        if show_interval:
            if interval_source == "Regression line":
                y_fit = model.predict(x_range.reshape(-1, 1))
                y_plus = y_fit * 1.2
                y_minus = y_fit * 0.8
            else:
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

        # --- Results below plot ---
        st.markdown(f"**Slope:** {slope_val:.4f}")
        if not through_origin:
            st.markdown(f"**Intercept:** {intercept_val:.4f}")
        else:
            st.markdown("**Intercept:** forced to 0")
        st.markdown(f"**R² score:** {r2:.4f}")
