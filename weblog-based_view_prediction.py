# Library Import
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read Datasets
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_sub = df_test['sessionID'].to_frame()

# Data concat for One-Hot Encoding
df = pd.concat([df_train, df_test])

# Column 정보 확인
col_list = df.columns.tolist()
print(col_list)
# df['browser'].value_counts().plot()
# df['browser'].value_counts(normalize=True)
# df['traffic_source'].isnull().sum()

# keyword + referral_path
def col_comb(col1, col2):
    if pd.isna(col1):
        if pd.isna(col2):
            result='(none)'
        else:
            result=col2.split('_')[0]
    else:
        result=col1.split('_')[0]
    return result

# Function for preprocessing
def preprocessing(df):
    # 미사용 컬럼 Drop
    df = df.drop('sessionID', axis=1)
    df = df.drop('userID', axis=1)
    df = df.drop('subcontinent', axis=1)
    
    # 비율이 적은 범주형 컬럼 값은 'Other'로 묶어서 대체함
    tmp_list = df['browser'].value_counts(normalize=True).loc[lambda x : x > 0.01].index.tolist()
    df['browser'] = df['browser'].apply(lambda x : x if x in tmp_list else 'Other')
    tmp_list = df['OS'].value_counts(normalize=True).loc[lambda x : x > 0.05].index.tolist()
    df['OS'] = df['OS'].apply(lambda x : x if x in tmp_list else 'Other')
    tmp_list = df['country'].value_counts(normalize=True).loc[lambda x : x > 0.01].index.tolist()
    df['country'] = df['country'].apply(lambda x : x if x in tmp_list else 'Other')
    tmp_list = df['traffic_source'].value_counts(normalize=True).loc[lambda x : x > 0.01].index.tolist()
    df['traffic_source'] = df['traffic_source'].apply(lambda x : x if x in tmp_list else 'Other')

    # 두 개의 컬럼의 Category 종류를 추출하여 새로운 컬럼으로 합침
    df['category'] = df.apply(lambda x: col_comb(x['keyword'], x['referral_path']), axis=1)
    df = df.drop(['keyword','referral_path'], axis=1)
    return df

# TARGET : 정답지
def TARGET(df):
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    return X, y


# 데이터 전처리
df = preprocessing(df)
# One-Hot Encoding
df = pd.get_dummies(df,columns=['browser','OS','device','continent','country','traffic_source','traffic_medium','category'])*1

# Train-Test 데이터 분리
df_train = df[df['TARGET'].notnull()]
df_test = df[df['TARGET'].isnull()]
df_test = df_test.drop('TARGET', axis=1)
print(df_train.shape)
print(df_test.shape)


# Train 데이터셋 X,y 분리
X_train, y_train = TARGET(df_train)

# Scailing
## Long-tail Data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(df_test)

# Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(model.intercept_)
print(model.coef_)

# Predict
pred = model.predict(X_test_scaled)
df_sub['TARGET']=pred
# df_sub.to_csv('result.csv', index=None)
