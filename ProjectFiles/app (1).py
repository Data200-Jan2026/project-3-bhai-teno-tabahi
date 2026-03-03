import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction using Property Features")
st.caption("California Housing Dataset — Aurélien Géron")

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_vif(X):
    vif_values = []
    for i in range(X.shape[1]):
        vif_values.append(variance_inflation_factor(X.values, i))
    return pd.DataFrame({"Feature": X.columns, "VIF": vif_values})

df = load_data(DATA_URL)

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

with col2:
    st.dataframe(df.head(10), use_container_width=True)

st.write("Missing Values:")
st.dataframe(df.isnull().sum().to_frame("Count"), use_container_width=True)

df_clean = df.dropna()

features = ["median_income", "total_rooms", "housing_median_age", "households"]
target = "median_house_value"

X = df_clean[features]
y = df_clean[target]

st.subheader("Preprocessing & Feature Selection")
st.write("Predictors:", features)
st.write("Target:", target)
st.write("Cleaned Data Shape:", df_clean.shape)

st.subheader("Multiple Linear Regression (OLS)")

test_ratio = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100

X_const = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_const, y, test_size=test_ratio, random_state=42
)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse_value = calculate_rmse(y_test, predictions)

m1, m2 = st.columns(2)

with m1:
    st.metric("R-squared", f"{r2:.4f}")

with m2:
    st.metric("RMSE", f"{rmse_value:,.2f}")

with st.expander("Regression Summary"):
    st.text(model.summary())

st.subheader("Model Diagnostics")

residuals = y_test - predictions

c1, c2 = st.columns(2)

with c1:
    fig1 = plt.figure()
    plt.scatter(predictions, residuals, s=10)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    st.pyplot(fig1)

with c2:
    fig2 = plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residuals")
    st.pyplot(fig2)

st.subheader("Multicollinearity (VIF)")
st.dataframe(compute_vif(X_const), use_container_width=True)

st.subheader("Quick EDA")

selected_feature = st.selectbox("Feature vs Price:", features)

e1, e2 = st.columns(2)

with e1:
    fig3 = plt.figure()
    plt.hist(df_clean[target], bins=40)
    plt.xlabel("Median House Value")
    st.pyplot(fig3)

with e2:
    fig4 = plt.figure()
    plt.scatter(df_clean[selected_feature], df_clean[target], s=8)
    plt.xlabel(selected_feature)
    plt.ylabel(target)
    st.pyplot(fig4)

st.subheader("Price Prediction")

ranges = {
    col: (float(df_clean[col].min()), float(df_clean[col].max()))
    for col in features
}

u1, u2 = st.columns(2)

with u1:
    income = st.slider("Median Income", *ranges["median_income"], float(df_clean["median_income"].median()))
    rooms = st.slider("Total Rooms", *ranges["total_rooms"], float(df_clean["total_rooms"].median()))

with u2:
    age = st.slider("Housing Median Age", *ranges["housing_median_age"], float(df_clean["housing_median_age"].median()))
    households = st.slider("Households", *ranges["households"], float(df_clean["households"].median()))

input_data = pd.DataFrame({
    "const": [1.0],
    "median_income": [income],
    "total_rooms": [rooms],
    "housing_median_age": [age],
    "households": [households],
})

predicted_price = model.predict(input_data)[0]

st.success(f"Predicted Median House Value: ${predicted_price:,.2f}")