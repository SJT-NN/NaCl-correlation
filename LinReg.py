import streamlit as st
import pandas as pd
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

    if x_col and y_col:
        # Prepare data
        X = df[[x_col]].dropna()
        y = df[y_col].dropna()

        # Align indices in case of missing values
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Build model
        if through_origin:
            model = LinearRegression(fit_intercept=False)
        else:
            model = LinearRegression()

        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="Data points")
        ax.plot(X, y_pred, color="red", linewidth=2, label="Regression line")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # --- Results ---
        st.markdown(f"**Slope:** {model.coef_[0]:.4f}")
        if not through_origin:
            st.markdown(f"**Intercept:** {model.intercept_:.4f}")
        else:
            st.markdown("**Intercept:** forced to 0")
        st.markdown(f"**R² score:** {r2:.4f}")
