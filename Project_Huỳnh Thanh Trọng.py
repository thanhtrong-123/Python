# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:51:48 2023

@author: trong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


data = pd.read_csv("Walmart.csv")
#thông tin tổng quan về dữ liệu
info = data.info()
print("\n")
print(data.head(5))
print("\n")
# Kiểm tra các giá trị còn thiếu trong mỗi cột
missing_values = data.isnull().sum()
print(missing_values) #Tập dữ liệu không có bất kỳ giá trị nào bị thiếu
print("\n")
# Kiểm tra giá trị trùng lặp
print(data.duplicated().sum()) # =0
print("\n")
#loại bỏ các cột không cần thiết
data.drop(['Date'], axis = 1, inplace = True)
print("\n")
#3 kiểu dữ liệu

print("Kieu du lieu")
data.dtypes
data['Store'] = data['Store'].astype('object')
data['Holiday_Flag'] = data['Holiday_Flag'].astype('object')
print(data.dtypes)
print("\n")
#Phát hiện và loại bỏ các ngoại lệ
print("Phát hiện và loại bỏ các ngoại lệ")
cols = ['Fuel_Price', 'Temperature', 'CPI', 'Unemployment']
plt.figure(figsize=(20,18))
for i,col in enumerate(cols):
    print(i, col)
    plt.subplot(3,2,i+1)
    sns.boxplot(data=data, x = col, color = 'red')
plt.show()

print('Number of data rows: ', data.shape[0])
print("\n")
# Chia dữ liệu thành dữ liệu huấn luyện và kiểm tra
X = data.drop('Weekly_Sales', axis = 1)
y = data['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('Shape of data      : ', X.shape)
print('Shape of train data: ', X_train.shape)
print('Shape of test data : ', X_test.shape)
print("\n")
# Chuyển đổi dữ liệu
# Chia dữ liệu thành Đặc điểm số và Đặc điểm phân loại
num_features = data.select_dtypes('number').columns.to_list()
num_features.remove('Weekly_Sales')
cat_features = data.select_dtypes('object').columns.to_list()
print(f'Numerical Features : {num_features}')
print(f'Categorical Features: {cat_features}')
print("\n")
#Tương quan dữ liệu
print("1. Giá nguyên liệu so với doanh số hàng tuần")
pearson_coef, p_value = stats.pearsonr(data['Fuel_Price'], data['Weekly_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef)
print("with a P-value of P =", p_value)
plt.figure(figsize = (14, 5))
sns.regplot(data = data, x = 'Fuel_Price', y = 'Weekly_Sales', color = '#145DA0', line_kws = {'color': 'red'})
plt.show()
print("\n")
print("2. Thất nghiệp so với doanh số hàng tuần")
pearson_coef, p_value = stats.pearsonr(data['Unemployment'], data['Weekly_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef)
print("with a P-value of P =", p_value)
plt.figure(figsize = (14, 5))
sns.regplot(data = data, x = 'Unemployment', y = 'Weekly_Sales', color = '#145DA0', line_kws = {'color': 'red'})
plt.show()
print("\n")
print("3. CPI so với doanh thu hàng tuần")
pearson_coef, p_value = stats.pearsonr(data['CPI'], data['Weekly_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef)
print("with a P-value of P =", p_value)
plt.figure(figsize = (14, 5))
sns.regplot(data = data, x = 'CPI', y = 'Weekly_Sales', color = '#145DA0', line_kws = {'color': 'red'})
plt.show()
print("\n")
print("4. Nhiệt độ so với doanh số hàng tuần")
pearson_coef, p_value = stats.pearsonr(data['Temperature'], data['Weekly_Sales'])
print("The Pearson Correlation Coefficient is", pearson_coef)
print("with a P-value of P =", p_value)
plt.figure(figsize = (14, 5))
sns.regplot(data = data, x = 'Temperature', y = 'Weekly_Sales', color = '#145DA0', line_kws = {'color': 'red'})
plt.show()
print("\n")
print("\n")

#Mô hình
#Hàm này dùng để đánh giá mô hình thông qua RMSE và R2
def model_evaluation(estimator, Training_Testing, X, y):
    # Y predict of X train or X test
    predict_data = estimator.predict(X)
    print(f'{Training_Testing} Accuracy:')
    print(f'-> Root Mean Squared Error: {round(np.sqrt(mean_squared_error(y, predict_data)), 2)}')
    print(f'-> R-Squere score Training: {round(r2_score(y, predict_data) * 100, 2)} % \n')

#Chức năng này được sử dụng để thực hiện một số đánh giá mô hình bằng cách sử dụng dữ liệu huấn luyện và kiểm tra
#bằng cách vẽ đồ thị phân bố các giá trị thực tế và dự đoán của dữ liệu huấn luyện hoặc kiểm tra.
def Distribution_Plot(estimator, Training_Testing, X, y, Title):
    # Y predict of X train or X test
    yhat = estimator.predict(X)
    plt.figure(figsize=(14, 6))
    # Use kdeplot for kernel density estimate
    sns.kdeplot(y, color="b", label=f'Actual Values ({Training_Testing})', fill=False)
    sns.kdeplot(yhat, color="r", label=f'Predicted Values ({Training_Testing})', fill=False)
    plt.title(Title, size=18)
    plt.legend()
    plt.show()

    
#Chức năng này là để xác nhận mô hình    
def cross_validation_score(estimator, X_train, y_train, score = 'r2', n = 5):
    validate = cross_val_score(estimator, X_train, y_train, scoring = score, cv = n)
    print(f'Độ lệch chuẩn của điểm: {validate.std()}')

#Chức năng này được sử dụng để tìm bộ siêu tham số tốt nhất cho mô hình nhằm tối ưu hóa hiệu suất của nó
def hyperparameter_tunning(estimator, X_train, y_train, param_grid, score = 'r2', n = 5):
    # Perform grid search
    grid_search = GridSearchCV(estimator = estimator,
                               param_grid = param_grid,
                               scoring = score,
                               cv = n)
    # Phù hợp với dữ liệu
    grid_search.fit(X_train,y_train)    
    # công cụ ước tính tốt nhất
    best_estimator = grid_search.best_estimator_
    return best_estimator

#1. Hồi quy tuyến tính
#1.1 Tạo mô hình (sử dụng LinearRegression()
LR = LinearRegression()
LR.fit(X_train, y_train)
#1.2 Điều chỉnh mô hình hồi quy tuyến tính với đa thức bậc cao 
LR_pipe = Pipeline([('poly_feat', PolynomialFeatures()),
                    ('lin_reg', LinearRegression())])
#Xác định lưới tham số để tìm kiếm
param_grid = {'poly_feat__degree': [2, 3, 4]}
best_estimator = hyperparameter_tunning(LR_pipe, X_train, y_train, param_grid, score = 'r2', n = 5)
poly_reg = best_estimator
#1.3 Đánh giá mô hình sử dụng cross-validation
cross_validation_score(poly_reg, X_train, y_train)
#1.4 Kiểm tra mô hình
print("Linear Regression:")
model_evaluation(poly_reg, 'Testing', X_test, y_test)
# Hình 1: Đồ thị giá trị dự đoán sử dụng dữ liệu thử nghiệm so với giá trị thực tế của dữ liệu thử nghiệm.
Title='Sơ đồ phân phối giá trị dự đoán bằng cách sử dụng dữ liệu thử nghiệm và phân phối dữ liệu của dữ liệu thử nghiệm'
Distribution_Plot(poly_reg, 'Testing', X_test, y_test, Title)
#Sau khi thử nghiệm mô hình hồi quy đa thức, đây  mô hình có độ chính xác của nó là 51.49%

#2 Công cụ hồi quy KNN
#2.1 Tạo mô hình
KNN_Reg = KNeighborsRegressor(n_neighbors = 5)
KNN_Reg.fit(X_train, y_train)
#2.3 Điều chỉnh mô hình
param_grid = {'n_neighbors': [1, 3, 5, 7, 8, 9, 11, 13]}
best_estimator = hyperparameter_tunning(KNN_Reg, X_train, y_train, param_grid, score = 'r2', n = 5)
Best_KNN = best_estimator
#2.5 Điểm xác thực chéo
cross_validation_score(Best_KNN, X_train, y_train, n = 10)
#2.6 Kiểm tra mô hình
print("KNN Regressor:")
model_evaluation(Best_KNN, 'Testing', X_test, y_test)
# Hình 3: Đồ thị giá trị dự đoán sử dụng dữ liệu thử nghiệm so với giá trị thực tế của dữ liệu thử nghiệm.
Title='Sơ đồ phân phối giá trị dự đoán bằng cách sử dụng dữ liệu thử nghiệm và phân phối dữ liệu của dữ liệu thử nghiệm'
Distribution_Plot(Best_KNN, 'Testing', X_test, y_test, Title)
# Kết Luận
#Sau khi thử nghiệm mô hình công cụ hồi quy KNN là mô hình khá tốt với độ chính xác của nó là 73.87%

#3 Cây quyết định hồi quy
#3.1 Tạo mô hình
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
#3.3 Điều chỉnh mô hình
param_grid = {'max_depth': np.arange(2, 15),
              'min_samples_split': [10, 20, 30, 40, 50, 100, 200, 300]}
best_estimator = hyperparameter_tunning(tree, X_train, y_train, param_grid, score = 'r2', n = 5)
Best_Tree = best_estimator
#3.5 Điểm xác thực chéo
cross_validation_score(Best_Tree, X_train, y_train, n = 10)
#3.6 Kiểm tra mô hình
print("Decision Tree Regressor:")
model_evaluation(Best_Tree, 'Testing', X_test, y_test)
# Figure 3: Plot of predicted value using the test data compared to the actual values of the test data.
Title='Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
Distribution_Plot(Best_Tree, 'Testing', X_test, y_test, Title)
# Kết Luận
#Sau khi thử nghiệm mô hình Hồi quy Cây quyết định là mô hình tốt nhất với độ chính xác của nó là 91.6%









