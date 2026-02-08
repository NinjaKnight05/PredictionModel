import streamlit as st
import joblib
import pickle
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.express as px
st.set_page_config(page_title='Customer Churn Prediction', page_icon='📉')

menu = option_menu(
    menu_title='',
    options=['Home', 'Prediction', 'Analysis'],
    icons=['house', 'activity', 'bar-chart'],
    orientation='horizontal'
)
df = pd.read_csv('Churn.csv')
model = joblib.load('churn_Data.pkl')
gendermod = pickle.load(open('gen.pkl', 'rb'))
geographymod = pickle.load(open('geo.pkl', 'rb'))

DEFAULT_SALARY = 100000  # change to your dataset mean if different


if menu == 'Home':
    st.title("🏦 Customer Churn Prediction System")

    st.markdown("""
    This application helps businesses **identify customers who are likely to leave (churn)** 
    using Machine Learning. It predicts churn risk in advance so companies can take 
    **preventive retention actions**.
    """)


    st.divider()

    st.warning("""
    ⚠️ **Disclaimer**  
    This model provides predictive insights, not guaranteed outcomes.  
    Predictions should be used to **support business decisions**, not replace human judgment.
    """)

    st.success("📌 Navigate to the **Prediction** section from the sidebar to get started.")


elif menu == 'Prediction':
    st.title('Customer Churn Prediction')

    credit_score = st.number_input( 'Credit Score', min_value=300,max_value=850,value=650)

    geography = st.selectbox('Geography', geographymod.classes_)
    geo_encoded = geographymod.transform([geography])[0]

    gender = st.selectbox('Gender', gendermod.classes_)
    gender_encoded = gendermod.transform([gender])[0]

    age = st.number_input('Age', min_value=18, max_value=100, value=40)
    tenure = st.number_input('Tenure (Years)', min_value=0, max_value=10, value=5)
    balance = st.number_input('Balance', min_value=0.0, value=50000.0)

    num_products = st.selectbox('Number of Products', [1, 2, 3, 4])

    has_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    has_card_val = 1 if has_card == 'Yes' else 0

    is_active = st.selectbox('Is Active Member', ['Yes', 'No'])
    is_active_val = 1 if is_active == 'Yes' else 0

    input_data = np.array([[
        credit_score,
        geo_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        has_card_val,
        is_active_val,
        DEFAULT_SALARY
    ]])

    if st.button('Predict'):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f'⚠️ High Churn Risk ({prob*100:.2f}%)')
            st.write('Recommended Action: Offer retention benefits or personalized support.')
        else:
            st.success(f'✅ Customer Likely to Stay ({(1-prob)*100:.2f}%)')

elif menu == 'Analysis':
    st.title('Model Analysis')
    st.write(df)
 
    st.subheader('CHURNED MEMBERS (M/F)')
    chart1=px.bar(df,y='Gender',x='Exited',color='Gender')
    st.write(chart1)
    
    st.subheader('Active Member And his Tenure')
    chart2 = px.pie(df,names='Tenure',values='IsActiveMember')
    st.plotly_chart(chart2)

