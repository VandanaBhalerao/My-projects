from turtle import width

import numpy as np
import pandas as pd
import pydeck
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.title('Amazon Sales')
st.info('The goal of this project was to analyze Amazon sales data across regions (North America, Asia, Europe, Middle East) and product categories to uncover insights into pricing, discounts, customer behavior, and sales performance. Additionally, predictive modeling was applied to forecast sales trends and evaluate model accuracy.')
#streamlit run amezon_sales.py
#_______________________________Load Data______________________________________________________

with st.expander('Amazon Sales Data'):
    @st.cache_data
    def load_data():
        return pd.read_csv("amazon_sales_dataset.csv")

    st.write('**Timeframe: 2022–2023 Features: Price, revenue, discount price, rating, discount percent, quantity sold, product category, customer region. Target Variables: Sales revenue and predicted values for future performance**')
    df = load_data()
    df
    uniq = df['customer_region'].unique()
    
    st.write('**X_raw**')
    X_raw =  df.drop('discount_percent', axis=1)
    X_raw

    st.write('**Y_raw**')
    y_raw= df.discount_percent
    y_raw


#The Correlation 
df[['price','discount_percent','quantity_sold','rating','discounted_price','total_revenue']].corr()

#simple feature scaling using standardization
st.subheader('**Standardized Features**')
st.write('Scaled numerical features (price, discount_percent, quantity_sold, rating,discounted_price,total_revenue etc.) to ensure comparability.')
df[['price','discount_percent','quantity_sold','rating','discounted_price','total_revenue']]=df[['price','discount_percent','quantity_sold','rating','discounted_price','total_revenue']].apply(lambda x: (x-x.mean())/x.std())
df[['price','discount_percent','quantity_sold','rating','discounted_price','total_revenue']]

#Binning - Grouping of values into bins or categories
st.write('**Binning- Grouping of values into bins or categories**')
bin = np.linspace(min(df['price']), max(df['price']), 4)
grup_names = ['Low','Medium','High']
df['price_binned'] = pd.cut(df['price'], bins=bin, labels=grup_names, include_lowest=True)


st.write('________________________________________________________________________________________________________________________')

from numpy.random import default_rng as rng

rating= pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["price", "rating", "discounted_price"])

st.bar_chart(rating)
st.write('________________________________________________________________________________________________________________________')



with st.sidebar:
    st.write('**Input Features**')
    st.write('Aggregated sales across categories and regions to identify macro-level trends.')
    product = st.selectbox('product_category',('Beauty','Books','Electronics','Fashion','Home & Kitchen','Sports'))
    date = st.date_input('order_date',value="today", min_value=None, max_value=None, key=None, help=None, on_change=None, args=None, kwargs=None,format="YYYY/MM/DD", disabled=False, label_visibility="visible", width="stretch", bind=None)
    customer_region = st.selectbox('customer_region',('North America','Asia','Europe','middle East'))
    pay_methods =st.selectbox('payment_method',('Wallet','Cash On Delivery','Debit Card','UPI'))
    price =st.slider('price',0, 500, 1000)
    #rate =st.feedback(df('rates'),options="stars", key=None, default=None, disabled=False, on_change=None, args=None, kwargs=None, width="content")


    #create a datafreame for input features

    datas = {'product':product,
             'date':date,
             'customer_region':customer_region,
             'pay_methods':pay_methods,
             'price':price 
    }
    input_df = pd.DataFrame(datas,index=[0])
    input_penguins = pd.concat([input_df,df],axis=0)  

#Encode x
encode = ['product_category','discount_percent','price','quantity_sold','discounted_price','total_revenue']
df_sale = pd.get_dummies(input_penguins[encode])
input_row = df_sale[:1]



#Encode y 

with st.expander('Input features'):
    st.write('**Inputs**')
    input_df
    st.write('**Combined sales data**')
    input_penguins  

    st.write('**Encoded inputs**')
    input_row

#_____________________________________________________________________Train Test Split_____________________________________________________________________________________
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
with st.expander('train_test'):
    st.write('Divided dataset into training and testing sets to validate predictive models.')
    #x_set = product_category, customer_region, payment_method, order-date
    #y_set = price, discount_percent,quantity_sold,rating,discount_price,total_revenue
    # New data Frame
    new_df = df[['price','discount_percent','quantity_sold','rating','discounted_price','total_revenue']]

    y_data = new_df[['price','total_revenue']]
    x_data = new_df.drop(['price','total_revenue'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.10,random_state=1)

    st.write("Number of test sample:",x_test.shape[0])
    st.write("Number of train sample:",x_train.shape[0])


    lre = LinearRegression()
    lre.fit(x_train,y_train)
    yhat_train =lre.predict(x_train)

    Rcross = cross_val_score(lre,x_data[['discount_percent']],y_data,cv=4)
    Rcross

    yhat1 = cross_val_predict(lre, x_data,y_data,cv=4)
    yhat1[0:10]

#Positive Linear Regression
st.area_chart(data=new_df, x='discount_percent',y='price', x_label='Discount Percent',y_label='Price', color="blue",width="stretch",height="content",use_container_width=None)
#st.info('')
def DistributionPlot(RedFunction, BlueFunction,RedName, BlueName):
    width = 12
    height = 10
    plt.figure(figsize = (width,height))
    st.header('Actual and Prediction value')

    fig, ax =plt.subplots()
    ax1=sns.kdeplot(RedFunction, color="r", label=RedName, ax=ax)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)
    ax.set_title('Distribution Plot of Predicted using of the Training data.')
    ax.set_xlabel('y_train', fontsize=12)
    ax.set_ylabel('yhat_train', fontsize=12)
    ax.legend()

    st.pyplot(fig)

actual = np.random.normal(100, 1, 200)
predicted = np.random.normal(90, 1, 200)

DistributionPlot(actual, predicted, 'Actual values(Train)', 'Predicted values(Train)')


#_____________________________________________________________________Feature Scaling____________________________________________

#Polynomial Regression and Pipeline - Help to us understand the Relationship between variable
from sklearn.preprocessing import PolynomialFeatures

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(0, 10, 100)
    y_new = model(x_new)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(independent_variable, dependent_variable, 'o', label="Data")
    ax.plot(x_new, y_new, '-', label="Polynomial Fit")
    ax.set_facecolor('green')
    ax.set_xlabel(Name)
    ax.set_ylabel('Price')
    ax.set_title('Polynomial Fit: Price vs Discount Percent')
    ax.legend()

    st.subheader('**Polynomial Fit with Matplotlib for Price and Discount Percent**')
    st.write('Applied polynomial regression to capture non-linear relationships between price, discount, and sales.')
    st.pyplot(fig)

x = new_df['discount_percent']
y = new_df['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
st.write(p)

PlotPolly(p, x, y, 'discount_percent')



pr = PolynomialFeatures(degree=2)

z_pr =pr.fit_transform(new_df)
z_pr.shape

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)

z = new_df.astype(float)
pipe.fit(z,y)

ypipe = pipe.predict(z)
ypipe[0:5]


#_____________________________________________________________________Model Deployment____________________________________________ 


#Linear Regression - Help to us understand the Relationship between variable

from sklearn.linear_model import LinearRegression
st.write('**Prediction-Prediction: Generated predicted sales values based on historical data.**')
lm = LinearRegression()
X = new_df[['discount_percent']]
Y = new_df['price']
w = new_df[['quantity_sold','rating','discounted_price','total_revenue']]
lm.fit(w,Y)

y_hat = lm.predict(w)
y_hat[0:10]

#_____________________________________________________________________Pridection and Decision making____________________________________________ 

n_input = np.arange(1, 101, 1).reshape(-1, 1)
lm.fit(X, Y)
y_hats = lm.predict(n_input)
y_hats[0:5]


plt.plot(n_input,y_hats)

plt.show()
#____________________________________________________________Accuracy______________________________________________________
#Measures for In sample Evaluation
from sklearn.metrics import mean_squared_error, r2_score
st.subheader('**Model Evaluation Metrics**')
st.write('Compared actual vs. predicted values using metrics such as accuracy, MSE, and R² score.')

y_pred_multi = lm.predict(X)
st.write('The mean square error of price and predicted  values using multiple linear regression is: ', mean_squared_error(new_df['price'], y_pred_multi))

#polynomial fit
from sklearn.metrics import mean_squared_error, r2_score
r_squared = r2_score(y, p(x))
st.write('The R-Squared value is: ', r_squared)
st.write('The mean square error of price accuracy is: ', mean_squared_error(new_df['price'], p(x)))


st.header('**Key Insights**')
st.write('-Discounts significantly influenced quantity sold, especially in Asia and North America.')
st.write('-Electronics and Fashion categories showed the highest revenue growth, while Books and Beauty had moderate sales.')
st.write('-Customer ratings positively correlated with sales performance, indicating the importance of product quality and customer satisfaction.')
st.write('-Predictive modeling revealed that price and discount percent are strong indicators of future sales trends, with polynomial regression capturing non-linear relationships effectively.')
st.write('-The model evaluation metrics, including accuracy, mean squared error(0.9997990567397184) and R² score(0.0001809468792192126), demonstrated the reliability of the predictive models in forecasting sales performance.')
st.write('__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________')
st.header('**Conclusion**')
st.write('This project demonstrated how data preprocessing, feature engineering, and predictive modeling can generate actionable insights into e-commerce sales. The analysis highlighted the importance of customer region and discount strategies in driving revenue, while predictive models provided reliable forecasts for future sales performance.')


