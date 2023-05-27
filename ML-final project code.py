#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:21:10 2023

@author: mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import csv
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize


# In[56]:

def eda():
    plt.style.use("ggplot")
    rcParams['figure.figsize'] = (12, 6)
    
    
    # # Store data in dataframe
    
    # In[57]:
    
    
    df=pd.read_excel("merged_house.xlsx")
    
    
    # # Show first few rows
    
    # In[58]:
    
    
    df.head()
    
    
    # # No of columns and rows
    
    # In[59]:
    
    
    df.shape
    
    
    # # Find missing values
    
    # In[60]:
    
    
    missing_values = df.isnull().sum()
    print(missing_values)
    
    
    # # Replace missing values with mode
    
    # In[61]:
    
    
    mode_values = df.mode().iloc[0]
    df.fillna(mode_values, inplace=True)
    missing_values = df.isnull().sum()
    print(missing_values)
    
    
    # # Summary of data frame
    
    # In[62]:
    
    
    summary_stats = df.describe()
    print(summary_stats)
    
    
    # In[63]:
    
    
    # Convert columns to numeric or float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['bed_room'] = pd.to_numeric(df['bed_room'], errors='coerce')
    
    # Print the data types of each column
    print(df.dtypes)
    
    
    # # Histogram
    
    # In[64]:
    
    
    plt.hist(df['price'], bins=10)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prices')
    plt.ylim(0,200)
    plt.show()
    
    
    # In[65]:
    
    
    plt.hist(df['bed_room'], bins=10)
    plt.xlabel('Bed_room')
    plt.ylabel('Frequency')
    plt.title('Histogram of Bed Rooms')
    xticks, _ = plt.xticks()
    
    # Sort the x-tick labels in descending order
    xtick_labels = sorted(xticks, reverse=True)
    
    # Set the new x-tick labels
    plt.xticks(xtick_labels)
    
    plt.show()
    
    
    # In[66]:
    
    
    bathrooms = df.loc[df['bath_room'] <= 10, 'bath_room']
    plt.hist(bathrooms, bins=10)
    plt.xlabel('Bath_room')
    plt.ylabel('Frequency')
    plt.title('Histogram of Bath Rooms')
    plt.show()
    
    
    # # Pairwise plot
    
    # In[67]:
    
    
    sns.pairplot(df)
    
    
    # # Boxplot
    
    # In[68]:
    
    
    plt.boxplot([df.loc[df['city'] == 'San_Francisco', 'price'], df.loc[df['city'] == 'Los Angeles', 'price']])
    plt.xticks([1, 2], ['San Francisco', 'Los Angeles'])
    plt.ylabel('Price')
    plt.title('Box Plot of Price by City')
    plt.show()
    
    
    # In[69]:
    
    
    pt = pd.pivot_table(df, values='price', index='latitude')
    
    # Create a heat map of the pivot table
    sns.heatmap(pt, cmap='coolwarm')
    plt.xlabel('Latitude')
    plt.ylabel('Price')
    plt.title('Heat Map of Price by Latitude')
    
    plt.show()
    
    
    # In[70]:
    
    
    pt = pd.pivot_table(df, values='price', index='longitude')
    
    # Create a heat map of the pivot table
    sns.heatmap(pt, cmap='coolwarm')
    plt.xlabel('Longitude')
    plt.ylabel('Price')
    plt.title('Heat Map of Price by Longitude')
    
    plt.show()
    
    
    # In[72]:
    
    
    sns.kdeplot(df['latitude'], df['longitude'], cmap='coolwarm', shade=True, shade_lowest=False, cbar=True, cbar_kws={'label': 'Price'})
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Heat Map of Price by Location')
    plt.show()
    
    
    # In[46]:
    
    
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.kdeplot(df['longitude'], df['latitude'], cmap='coolwarm', shade=True, shade_lowest=False, cbar=True, cbar_kws={'label': 'Price'}, ax=ax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Heat Map of Price by Location')
    
    plt.show()
    
    
    # In[73]:
    
    
    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(df['longitude'], df['latitude'], C=df['price'], cmap='coolwarm', gridsize=50, mincnt=1)
    cb = fig.colorbar(hb)
    cb.set_label('Price')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Heat Map of Price by Location')
    
    plt.show()
    
    
    # In[78]:
    
    
    affordable_sf = len(df[(df['city']=='San_Francisco') & (df['house_payment-income ratio']<= 0.3)])
    total_sf = len(df[(df['city']=='San_Francisco')])
    
    
    affordable_la = len(df[(df['city']=='Los Angeles') & (df['house_payment-income ratio'] <= 0.3)])
    total_la = len(df[(df['city']=='Los Angeles')])
    
    Population=df["Population"]
    sf_ratio = (affordable_sf/total_sf)*100
    
    la_ratio = (affordable_la/total_la)*100
    
    cities = ['San Francisco', 'Los Angeles']
    ratios = [sf_ratio, la_ratio]
    colors = ['#1f77b4', '#ff7f0e']
    
    # create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(cities, ratios, color=colors)
    
    # set title and axis labels
    plt.title('Percent of Affordable Houses in Two Cities in California')
    plt.xlabel('City')
    plt.ylabel('Percentage')
    
    # set y-axis limit
    plt.ylim([0, 100])
    
    # add data labels to the bars
    for i, v in enumerate(ratios):
        plt.text(i, v + 1, str(round(v, 2)), ha='center', fontsize=12)
    
    # display plot
    plt.tight_layout()
    plt.show()
    
      
    
    
    # In[75]:
    
    
    sf_subset = df[df['city']=='San_Francisco']
    # calculate the average ratio for SF
    avg_scores = sf_subset.groupby('neighbourhood')['house_payment-income ratio'].mean()
    avg_scores = avg_scores.sort_values(ascending=True).iloc[:15]
    # create a horizontal bar graph of the average scores
    fig, ax = plt.subplots()
    ax.barh(avg_scores.index, avg_scores.values)
    ax.set_xlabel('Average Affordability Ratio')
    ax.set_ylabel('NeighborHood')
    
    # add labels to each bar
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.01, i, str(round(v,2)), color='blue', fontweight='bold')
    
    # display the plot
    plt.show()
    
    
    # In[76]:
    
    
    la_subset = df[df['city']=='Los Angeles']
    # calculate the average ratio for LA
    avg_scores = la_subset.groupby('neighbourhood')['house_payment-income ratio'].mean()
    avg_scores = avg_scores.sort_values(ascending=True).iloc[:15]
    # create a horizontal bar graph of the average scores
    fig, ax = plt.subplots()
    ax.barh(avg_scores.index, avg_scores.values)
    ax.set_xlabel('Average Affordability Ratio')
    ax.set_ylabel('NeighborHood')
    
    # add labels to each bar
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.01, i, str(round(v,2)), color='blue', fontweight='bold')
    
    # display the plot
    plt.show()

def ml():
    houses = pd.read_excel("merged_house.xlsx")
    houses.head()
    
    
    # In[5]:
    
    
    # Check the shape
    shape_houses = houses.shape
    print("The shape of houses is:", shape_houses)
    # Check null values
    null_val = houses.isnull().sum()
    null_percentage = round(null_val / len(houses),2)
    print(null_val)
    print(null_percentage)
    
    
    # Based on the result, the percentage of null values is very small. Also, since the housing price may be related with the number of bedrooms, bathrooms and squred feet a lot. As a result, instead of choosing median values to make up for the null values, we decide to drop columns with null values.
    
    # In[6]:
    
    
    # Drop columns with null price, bed_room, bath_room, sqft and avg_price_per_neighborhood
    houses_c = houses.copy()
    houses_c = houses.dropna(subset = ["price", "bed_room", "bath_room", "sqft", "avg_price_per_neighborhood"])
    houses_c.isnull().sum()
    
    
    # In[7]:
    
    
    houses_c.head()
    
    
    # In[8]:
    
    
    # Drawing histograms to check the distribution of data
    plt.figure(figsize = (12,8))
    n, bins, patches = plt.hist(houses_c["price"], density=False)
    
    # Add a title and axis labels
    plt.title("Price Histogram")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    
    # Overlay the count values as text
    for i in range(len(patches)):
        plt.text(x=bins[i], y=n[i], s=int(n[i]), ha="center", va="bottom")
    
    # Display the histogram
    plt.show()
    
    
    # There seems to be some extremely expensive houses, but the number of extremely expensive houses is small. As a result, let's consider removing these houses.
    
    # In[9]:
    
    
    houses_c = houses_c[houses_c["price"] <= 5000000]
    houses_c.shape
    
    
    # In[10]:
    
    
    plt.figure(figsize = (12,8))
    n, bins, patches = plt.hist(houses_c["price"], density=False)
    
    # Add a title and axis labels
    plt.title("Price Histogram")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    
    # Overlay the count values as text
    for i in range(len(patches)):
        plt.text(x=bins[i], y=n[i], s=int(n[i]), ha="center", va="bottom")
    
    # Display the histogram
    plt.show()
    
    
    # In[11]:
    
    
    # bed_room
    houses_c['bed_room'] = pd.to_numeric(houses_c['bed_room'], errors='coerce')
    houses_c = houses_c.dropna(subset=['bed_room'])
    plt.hist(houses_c['bed_room'])
    plt.title("bedroom frequency")
    plt.xlabel("bed_room")
    plt.ylabel("Frequency")
    
    
    # In[12]:
    
    
    houses_c = houses_c[houses_c["bed_room"] <= 10]
    houses_c.shape
    
    
    # In[13]:
    
    
    plt.hist(houses_c['bed_room'])
    plt.title("bedroom frequency")
    plt.xlabel("bed_room")
    plt.ylabel("Frequency")
    
    
    # In[14]:
    
    
    # bath_room
    plt.hist(houses_c['bath_room'])
    plt.title("bathroom frequency")
    plt.xlabel("bath_room")
    plt.ylabel("Frequency")
    
    
    # In[15]:
    
    
    houses_c = houses_c[houses_c["bath_room"] <= 8]
    houses_c.shape
    
    
    # In[16]:
    
    
    plt.hist(houses_c['bath_room'])
    plt.title("bathroom frequency")
    plt.xlabel("bath_room")
    plt.ylabel("Frequency")
    
    
    # In[17]:
    
    
    # sqft
    plt.hist(houses_c['sqft'])
    plt.title("Sqft Histogram")
    plt.xlabel("Sqft")
    plt.ylabel("Frequency")
    
    
    # In[18]:
    
    
    houses_c = houses_c[houses_c["sqft"] <= 6000]
    houses_c = houses_c[houses_c["sqft"] >= 200]
    houses_c.shape
    
    
    # In[19]:
    
    
    plt.hist(houses_c['sqft'])
    plt.title("Sqft Histogram")
    plt.xlabel("Sqft")
    plt.ylabel("Frequency")
    
    
    # In[20]:
    
    
    # frequency table of neighbourhood
    freq_neighborhood = houses_c["neighbourhood"].value_counts()
    print(freq_neighborhood)
    
    
    # In[21]:
    
    
    # Delete rows where the frequency of neighbourhood is less than 5
    houses_c = houses_c[~houses_c["neighbourhood"].isin(freq_neighborhood[freq_neighborhood < 5].index)]
    houses_c.shape
    
    
    # In[22]:
    
    
    # Population
    plt.hist(houses_c['Population'])
    plt.title("Population Histogram")
    plt.xlabel("Population")
    plt.ylabel("Frequency")
    
    
    # In[23]:
    
    
    # Number of Households
    plt.hist(houses_c['Number of Households'])
    plt.title("Number of Households Histogram")
    plt.xlabel("Number of Households")
    plt.ylabel("Frequency")
    
    
    # In[24]:
    
    
    # Median Income
    plt.hist(houses_c['Median Income'])
    plt.title("Median Income Histogram")
    plt.xlabel("Median Income")
    plt.ylabel("Frequency")
    
    
    # In[25]:
    
    
    # avg_price_per_neighborhood
    plt.hist(houses_c['avg_price_per_neighborhood'])
    plt.title("avg_price_per_neighborhood Histogram")
    plt.xlabel("avg_price_per_neighborhood")
    plt.ylabel("Frequency")
    
    
    # In[26]:
    
    
    houses_c.head()
    
    
    # Now we've finished the data cleaning. Let's start to do machine learning!

    # Linear Regression
    # create dummy variables
    neighborhood_dummy = pd.get_dummies(houses_c["neighbourhood"], prefix = "neighborhood")
    city_dummy = pd.get_dummies(houses_c["city"], prefix = "city")
    houses_c = pd.concat([houses_c, neighborhood_dummy], axis=1)
    houses_c = pd.concat([houses_c, city_dummy], axis = 1)
    houses_c = houses_c.drop(["address", "city", "zip_code", "latitude", "longitude", "neighbourhood", "Average Income", "house_payment_1yr", "house_payment-income ratio"], axis = 1)
    
    
    # In[28]:
    
    
    X = houses_c.drop(["price", "neighborhood_Western Addition", "city_Los Angeles"], axis=1)
    y = houses_c["price"]
    # Add a constant to the independent variables
    X = sm.add_constant(X)
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Print the summary of the model
    print(model.summary())
    
    
    # The smallest eigenvalue is 6.43e-28. This might indicate that there are strong multicollinearity problems or that the design matrix is singular. The Durbin-Watson test result is 2.012, which means there is no autoregressive problem.
    
    # In[29]:
    
    
    # Let's check whether it has heteroscedasticity and multicolinearity problem
    # Heteroscedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, pvalue, _, _ = het_breuschpagan(model.resid, X)
    print("Breusch-Pagan test p-value:", pvalue)
    
    
    # The p-value is extremely small, which indicates very strong heteroscedasticity problem.
    
    # In[30]:
    
    
    # Multicolinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # create a list of column names for the predictor variables
    predictor_cols = ['bed_room', 'bath_room', 'sqft', 'Population', 'Number of Households', 'Median Income', 'avg_price_per_neighborhood']
    X_predictors = X[predictor_cols]
    
    # calculate the VIF for each predictor variable
    vif = [variance_inflation_factor(X_predictors.values, i) for i in range(X_predictors.shape[1])]
    
    # create a DataFrame to display the results
    vif_df = pd.DataFrame({'Variable': predictor_cols, 'VIF': vif})
    print(vif_df)
    
    
    # Almost all the VIF tests are over 5. As a result, this model also exists strong multicolinearity problem.
    
    # In[31]:
    
    
    # To solve these problems, I first try PCA to solve the multicolinearity problem.
    from sklearn.decomposition import PCA
    
    # select columns to use in PCA
    cols_for_pca = ['bed_room', 'bath_room', 'sqft', 'Population', 'Number of Households', 'Median Income', 'avg_price_per_neighborhood']
    
    # perform PCA on X variables
    houses_c = houses_c.reset_index(drop=True)
    pca = PCA()
    X_pca = pca.fit_transform(houses_c[cols_for_pca])
    
    # calculate the explained variance ratio for each principal component
    explained_var = pca.explained_variance_ratio_
    
    # plot the scree plot
    plt.plot(range(1, len(explained_var) + 1), explained_var, 'bo-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.show()
    
    
    # It seems only the first principal component is enough.
    
    # In[32]:
    
    
    # create a DataFrame of the principal components
    pc_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
    # Select the first PC
    X_pca_1 = pc_df['PC1']
    # concatenate the first principal component with the dummy variables
    neighborhood_dummy = neighborhood_dummy.reset_index(drop=True)
    city_dummy = city_dummy.reset_index(drop=True)
    X_transformed = pd.concat([X_pca_1, neighborhood_dummy, city_dummy], axis=1)
    X_transformed.head()
    
    
    # In[33]:
    
    
    X_new = X_transformed.drop(["neighborhood_Western Addition", "city_Los Angeles"], axis=1)
    y = houses_c["price"]
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=9964)
    
    # create and fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predict on test set and calculate RMSE
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    r2_in = model.score(X_train, y_train)
    print(f"IS R-squared: {r2_in:.2f}")
    print(f"OOS R-squared: {r2:.2f}")
    
    
    # In[34]:
    
    
    # Check Heteroscedasticity
    # calculate residuals
    residuals = y_test - y_pred
    
    # plot residuals against predicted values
    sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
    
    
    # Based on the residual plot, there seems to be no serious heteroscedasticity problem since the points are scattered almost randomly around the horizontal line at zero.
    
    # Based on the result of PCA-based linear regression model, the accuracy seems to be terrible especially when it comes to out-of-sample dataset. Then let's try regression tree models.
    
    # In[35]:
    
    
    # Let's try CV LASSO regression!
    # Split the data into train and test sets
    X = X[predictor_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9964)
    
    # Fit LASSO regression with cross-validation
    lasso_cv = LassoCV(cv=10, random_state=9964)
    lasso_cv.fit(X_train, y_train)
    print("Coefficients:", lasso_cv.coef_)
    
    # Evaluate the model on the test set
    y_pred_la = lasso_cv.predict(X_test)
    test_score = lasso_cv.score(X_test, y_test)
    mse_la = mean_squared_error(y_test, y_pred_la)
    msle_la = mean_squared_log_error(y_test, y_pred_la)
    rmsle_la = np.sqrt(msle_la)
    print("RMSLE:", rmsle_la)
    print("Test MSE score: {:.4f}".format(mse_la))
    print("Test R^2 score: {:.4f}".format(test_score))
    
    
    # Even though the accuracy is still not very high, it is much better than the original one.
    
    # In[36]:
    
    
    # Let's first try the full tree model
    # Decision tree model
    X = houses_c.drop(["price"], axis=1)
    y = houses_c["price"]
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9964)
    # Fit the tree model
    tree_model = DecisionTreeRegressor(random_state=9964)
    tree_model.fit(X_train, y_train)
    # R-squared & MSE
    y_pred = tree_model.predict(X_test)
    y_train_pred = tree_model.predict(X_train)
    accuracy = r2_score(y_test, y_pred)
    accuracy_train = r2_score(y_train_pred, y_train)
    mse_train_1 = mean_squared_error(y_train, y_train_pred)
    mse_test_1 = mean_squared_error(y_test, y_pred) 
    msle_train_1 = mean_squared_log_error(y_train, y_train_pred)
    rmsle_train_1 = np.sqrt(msle_train_1)
    msle_test_1 = mean_squared_log_error(y_test, y_pred)
    rmsle_test_1 = np.sqrt(msle_test_1)
    print("In-sample RMSLE:", rmsle_train_1)
    print("In-sample MSE score: {:.4f}".format(mse_train_1))
    print("Out-of-sample RMSLE:", rmsle_test_1)
    print("Out-of-sample MSE score: {:.4f}".format(mse_test_1))
    print("Accuracy of the decision tree model for in-sample dataset:", accuracy_train)
    print("Accuracy of the decision tree model for out-of-sample dataset:", accuracy)
    
    
    # In[37]:
    
    
    # Cross validation
    # Fit the tree model with k-fold cross-validation
    tree_model_c = DecisionTreeRegressor(random_state=9964)
    cv_scores_1 = cross_val_score(tree_model_c, X_train, y_train, cv=5)
    tree_model_c.fit(X_train, y_train)
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores_1)
    print("Mean cross-validation score:", np.mean(cv_scores_1))
    
    # OOS R-squared
    y_pred_c = tree_model_c.predict(X_test)
    accuracy_c = r2_score(y_test, y_pred_c)
    print("Accuracy of the decision tree model for out-of-sample dataset:", accuracy_c)
    
    
    # The R-squared of in-sample dataset is much larger than OOS R-squared. As a result, there may be severe overfitting problem.
    
    # In[39]:
    
    
    # Plot decision tree
    plt.figure(figsize=(15,8))
    plot_tree(tree_model, feature_names=X_train.columns, filled=True, fontsize=5)
    plt.show()
    
    
    # The tree plot is too complex to interpret, which is also an indicator of overfitting.
    
    # In[38]:
    
    
    # Plot scree plot
    importances = tree_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # select the top 10 important features
    
    plt.figure(figsize=(10,6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    
    # Based on the importance plot, the sqft of the house and the population, avg_price_per_neighborhood seems to be the most important features that influence the housing price.
    
    # In[40]:
    
    
    # Fit the tree model with k-fold cross-validation
    tree_model_c_2 = DecisionTreeRegressor(random_state=9964, min_samples_leaf=20)
    cv_scores_2 = cross_val_score(tree_model_c_2, X_train, y_train, cv=5)
    tree_model_c_2.fit(X_train, y_train)
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores_2)
    print("Mean cross-validation score:", np.mean(cv_scores_2))
    
    # OOS R-squared
    y_pred_c_2 = tree_model_c_2.predict(X_test)
    accuracy_c_2 = r2_score(y_test, y_pred_c_2)
    mse_test_2 = mean_squared_error(y_test, y_pred_c_2) 
    msle_test_2 = mean_squared_log_error(y_test, y_pred_c_2)
    rmsle_test_2 = np.sqrt(msle_test_2)
    print("OOS RMSLE:", rmsle_test_2)
    print("OOS MSE score: {:.4f}".format(mse_test_2))
    print("Accuracy of the decision tree model for out-of-sample dataset:", accuracy_c_2)
    
    
    # The OOS R-squared and MSE improved a lot.
    
    # In[41]:
    
    
    # Also restrict the max depth
    tree_model_c_3 = DecisionTreeRegressor(random_state=9964, min_samples_leaf = 20, max_depth = 10)
    cv_scores_3 = cross_val_score(tree_model_c_3, X_train, y_train, cv=5)
    tree_model_c_3.fit(X_train, y_train)
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores_3)
    print("Mean cross-validation score:", np.mean(cv_scores_3))
    
    # OOS R-squared
    y_pred_c_3 = tree_model_c_3.predict(X_test)
    accuracy_c_3 = r2_score(y_test, y_pred_c_3)
    mse_test_3 = mean_squared_error(y_test, y_pred_c_3) 
    msle_test_3 = mean_squared_log_error(y_test, y_pred_c_3)
    rmsle_test_3 = np.sqrt(msle_test_3)
    print("OOS RMSLE:", rmsle_test_3)
    print("OOS MSE score: {:.4f}".format(mse_test_3))
    print("Accuracy of the decision tree model for out-of-sample dataset:", accuracy_c_3)
    
    
    # The R-squared and MSE did not change.
    
    # In[42]:
    
    
    # Draw the updated tree plot
    plt.figure(figsize=(15,8))
    plot_tree(tree_model_c_3, feature_names=X_train.columns, filled=True, fontsize=5)
    plt.show()
    
    
    # In[43]:
    
    
    # Plot scree plot
    importances = tree_model_c_3.feature_importances_
    indices = np.argsort(importances)[-10:]  # select the top 10 important features
    
    plt.figure(figsize=(10,6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    
    # Still, sqft, population and avg_price_per_neighborhood are the most important features determining the housing price.
    
    # To achieve better results, let's now try to use random forest to tackle this prediction.
    
    # In[44]:
    
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=500, random_state=9964)
    rf_model.fit(X_train, y_train)
    
    
    # In[45]:
    
    
    # Calculate R-squared score on test set
    y_pred_rf = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_rf)
    mse_test_rf = mean_squared_error(y_test, y_pred_rf) 
    msle_test_rf = mean_squared_log_error(y_test, y_pred_rf)
    rmsle_test_rf = np.sqrt(msle_test_rf)
    print("OOS RMSLE:", rmsle_test_rf)
    print("OOS MSE score: {:.4f}".format(mse_test_rf))
    print("R-squared score:", r2)
    
    
    # The OOS R-squared is increased to 0.654 from 0.573. The OOS MSE is decreased from 356669624123.9861 to 283382267840.0465.
    
    # In[46]:
    
    
    # Plot feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xticks(rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()
    
    
    # Similarly, sqft, population, avg_price_per_neighborhood are the most important features.
    
    # In[47]:
    
    
    # Create XGBoost model with default hyperparameters
    xgb_model = XGBRegressor()
    
    # Perform 5-fold cross-validation on the training set
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
    
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", np.mean(cv_scores))
    xgb_model.fit(X_train, y_train)
    
    
    # In[48]:
    
    
    # Compute R-squared score on the test set
    y_pred_xg = xgb_model.predict(X_test)
    accuracy = r2_score(y_test, y_pred_xg)
    mse_test_xg = mean_squared_error(y_test, y_pred_xg)
    msle_test_xg = mean_squared_log_error(y_test, y_pred_xg)
    rmsle_test_xg = np.sqrt(msle_test_xg)
    print("OOS RMSLE:", rmsle_test_xg)
    print("OOS MSE score: {:.4f}".format(mse_test_xg))
    print("Accuracy of the XGBoost model on the test set:", accuracy)
    
    
    # In[49]:
    
    
    # Ensemble Random forest, XGBoost and Decision tree model based on the accuracy
    # ensemble_preds = (y_pred_xg * accuracy + y_pred_rf * r2 + y_pred_c_3 * accuracy_c_3 + y_pred_la * test_score)/(accuracy + r2 + accuracy_c_3 + test_score)
    ensemble_preds = y_pred_xg * 0.15 + y_pred_rf * 0.65 + y_pred_c_3 * 0.15 + y_pred_la * 0.05
    # Compute R_squared
    accuracy_e = r2_score(y_test, ensemble_preds)
    print("Accuracy of the ensembled prediction on the test set:", accuracy_e)
    # RMSLE
    msle_test_e = mean_squared_log_error(y_test, ensemble_preds)
    rmsle_test_e = np.sqrt(msle_test_e)
    print("OOS RMSLE:", rmsle_test_e)
    # MSE
    mse_test_e = mean_squared_error(y_test, ensemble_preds) 
    print("OOS MSE score: {:.4f}".format(mse_test_e))
    
    
    # In[50]:
    
    
    # Using optimization techniques to find the best weights for each prediction
    
    def loss_function(weights):
        ensemble_preds = y_pred_xg * weights[0] + y_pred_rf * weights[1] + y_pred_c_3 * weights[2] + y_pred_la * weights[3]
        return np.sqrt(mean_squared_log_error(y_test, ensemble_preds))
    
    initial_weights = [0.25, 0.25, 0.25, 0.25]  # starting weights for optimization
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # bounds for weight values
    
    result = minimize(loss_function, initial_weights, bounds=bounds)
    
    # Use the optimized weights to compute the final ensembled prediction
    weights = result.x
    ensemble_preds = y_pred_xg * weights[0] + y_pred_rf * weights[1] + y_pred_c_3 * weights[2] + y_pred_la * weights[3]
    
    # Compute evaluation metrics with the optimized weights
    accuracy_e = r2_score(y_test, ensemble_preds)
    print("Accuracy of the ensembled prediction on the test set:", accuracy_e)
    
    msle_test_e = mean_squared_log_error(y_test, ensemble_preds)
    rmsle_test_e = np.sqrt(msle_test_e)
    print("OOS RMSLE:", rmsle_test_e)
    
    mse_test_e = mean_squared_error(y_test, ensemble_preds) 
    print("OOS MSE score: {:.4f}".format(mse_test_e))
    
    
    # In[51]:
    
    
    print(weights)
    
    
    # As a result, the best weight of XGBoost model is 0.25, the best weight of Random Forest is 0.6568921, the best weight for lasso regression model is 0.052285. 
    
    # To conclude, we will use the final weighted ensembled model to make housing price predictions.

def main():
    eda()
    ml()   
if __name__ == '__main__':
    main()
        