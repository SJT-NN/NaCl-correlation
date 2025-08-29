import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📊 Excel Correlation & Regression Visualizer")
st.text("The code can be found on https://github.com/SJT-NN?tab=repositories")

# --- Upload Excel file ---
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    # Get sheet names
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    # Select sheet
    sheet_name = st.selectbox("Select sheet", sheet_names)

    # Load selected sheet
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    st.subheader(f"Preview of Data — Sheet: {sheet_name}")
    st.dataframe(df.head())

    # --- Column selectors ---
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)
    yerr_col = st.selectbox("Select Y-error column (optional)", ["None"] + cols)
    category_col = st.selectbox("Select Category column for coloring (optional)", ["None"] + cols)

    # --- Color palette selector ---
    palette_source = st.selectbox(
        "Choose color palette source",
        ["Matplotlib tab20", "Seaborn deep", "Seaborn Set3", "ColorCET glasbey"]
    )

    # --- Axis label inputs ---
    custom_x_label = st.text_input("Custom X-axis label", value=x_col)
    custom_y_label = st.text_input("Custom Y-axis label", value=y_col)
    custom_title = st.text_input("Custom title", value="")

    # --- Display options ---
    point_size = st.slider("Scatter point size", 10, 200, 50)
    plot_width = st.slider("Plot width", 4, 16, 8)
    plot_height = st.slider("Plot height", 4, 12, 6)

    # --- Analysis options ---
    through_origin = st.checkbox("Force regression through (0,0)")
    show_interval = st.checkbox("Show ±20% interval in green")
    interval_source = st.selectbox("Interval source", ["Regression line", "y = x identity line"])

    if x_col and y_col:
        required_cols = [x_col, y_col]
        if yerr_col != "None":
            required_cols.append(yerr_col)
        if category_col != "None":
            required_cols.append(category_col)

        df_valid = df[required_cols].dropna()

        for col in [x_col, y_col] + ([] if yerr_col == "None" else [yerr_col]):
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
        df_valid = df_valid.dropna()

        if df_valid.empty:
            st.error("No valid numeric rows found after cleaning. Check your column selections.")
        else:
            # --- Optional category filtering ---
            if category_col != "None":
                categories = sorted(df_valid[category_col].astype(str).unique())
                selected_categories = st.multiselect(
                    f"Select {category_col} values to include",
                    options=categories,
                    default=categories
                )
                df_filtered = df_valid[df_valid[category_col].astype(str).isin(selected_categories)]
            else:
                df_filtered = df_valid.copy()

            if not df_filtered.empty:
                X = df_filtered[[x_col]].values
                y = df_filtered[y_col].values
                yerr = df_filtered[yerr_col].values if yerr_col != "None" else None

                # Axis limits
                x_data_min, x_data_max = float(df_filtered[x_col].min()), float(df_filtered[x_col].max())
                y_data_min, y_data_max = float(df_filtered[y_col].min()), float(df_filtered[y_col].max())

                if st.checkbox("Expand axis limits by ±1", value=False):
                    x_data_min -= 1; x_data_max += 1
                    y_data_min -= 1; y_data_max += 1

                xlim_min, xlim_max = st.slider("X-axis range",
                    min_value=float(x_data_min), max_value=float(x_data_max),
                    value=(float(x_data_min), float(x_data_max)), step=0.1
                )
                ylim_min, ylim_max = st.slider("Y-axis range",
                    min_value=float(y_data_min), max_value=float(y_data_max),
                    value=(float(y_data_min), float(y_data_max)), step=0.1
                )

                # Regression fit
                model = LinearRegression(fit_intercept=not through_origin)
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                slope_val = model.coef_[0]
                intercept_val = model.intercept_ if not through_origin else 0

                x_min, x_max = X.min(), X.max()
                x_range = np.linspace(x_min, x_max, 500)
                sorted_idx = np.argsort(X.flatten())

                fig, ax = plt.subplots(figsize=(plot_width, plot_height))

                # Build color list
                if category_col != "None":
                    if palette_source.startswith("Matplotlib"):
                        colors = plt.get_cmap("tab20").colors
                    elif palette_source.startswith("Seaborn"):
                        sns_name = palette_source.split(" ")[1]
                        colors = sns.color_palette(sns_name, n_colors=len(selected_categories))
                    elif palette_source.startswith("ColorCET"):
                        colors = list(cc.glasbey)

                    for idx, cat in enumerate(selected_categories):
                        mask = df_filtered[category_col].astype(str) == cat
                        color = colors[idx % len(colors)]
                        if yerr is not None:
                            ax.errorbar(
                                X[mask].flatten(), y[mask], yerr=yerr[mask],
                                fmt='o', alpha=0.7, markersize=point_size/10,
                                label=str(cat), color=color
                            )
                        else:
                            ax.scatter(
                                X[mask], y[mask],
                                s=point_size, alpha=0.7,
                                label=str(cat), color=color
                            )
                    ax.legend(title=category_col)
                else:
                    if yerr is not None:
                        ax.errorbar(
                            X.flatten(), y, yerr=yerr,
                            fmt='o', alpha=0.7, markersize=point_size/10,
                            label="Data points with error"
                        )
                    else:
                        ax.scatter(
                            X, y, s=point_size, alpha=0.7,
                            label="Data points"
                        )

                # Regression line
                ax.plot(
                    X[sorted_idx], y_pred[sorted_idx],
                    color="red", linewidth=2,
                    label=f"Regression line (slope={slope_val:.4f}, intercept={intercept_val:.4f}, R²={r2:.4f})"
                )

                # y = x line
                ax.plot(x_range, x_range, color="gray", linestyle="--", label="y = x")

                # ±20% interval
                if show_interval:
                    if interval_source == "Regression line":
                        y_fit = model.predict(x_range.reshape(-1, 1))
                        y_plus, y_minus = y_fit * 1.2, y_fit * 0.8
                    else:
                        y_plus, y_minus = x_range * 1.2, x_range * 0.8

                    ax.fill_between(x_range, y_minus, y_plus, color="green", alpha=0.2,
                                    label=f"±20% from {interval_source.lower()}")

                ax.set_xlabel(custom_x_label)
                ax.set_ylabel(custom_y_label)
                ax.set_title(custom_title)
                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylim(ylim_min, ylim_max)
                ax.grid(True)
                ax.legend(loc='center right',bbox_to_anchor=(1.5,0.5),fontsize=8)

                st.pyplot(fig)

                st.markdown(f"**Slope:** {slope_val:.4f}")
                st.markdown(f"**Intercept:** {intercept_val:.4f}" if not through_origin else "**Intercept:** forced to 0")
                st.markdown(f"**R² score:** {r2:.4f}")
