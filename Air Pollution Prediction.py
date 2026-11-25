{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b6e266e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "import statsmodels.api as sm\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "34c5a7f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Entity</th>\n",
       "      <th>Code</th>\n",
       "      <th>Year</th>\n",
       "      <th>Air pollution (total) (deaths per 100,000)</th>\n",
       "      <th>Indoor air pollution (deaths per 100,000)</th>\n",
       "      <th>Outdoor particulate matter (deaths per 100,000)</th>\n",
       "      <th>Outdoor ozone pollution (deaths per 100,000)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>AFG</td>\n",
       "      <td>1990</td>\n",
       "      <td>299.477309</td>\n",
       "      <td>250.362910</td>\n",
       "      <td>46.446589</td>\n",
       "      <td>5.616442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>AFG</td>\n",
       "      <td>1991</td>\n",
       "      <td>291.277967</td>\n",
       "      <td>242.575125</td>\n",
       "      <td>46.033841</td>\n",
       "      <td>5.603960</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>AFG</td>\n",
       "      <td>1992</td>\n",
       "      <td>278.963056</td>\n",
       "      <td>232.043878</td>\n",
       "      <td>44.243766</td>\n",
       "      <td>5.611822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>AFG</td>\n",
       "      <td>1993</td>\n",
       "      <td>278.790815</td>\n",
       "      <td>231.648134</td>\n",
       "      <td>44.440148</td>\n",
       "      <td>5.655266</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>AFG</td>\n",
       "      <td>1994</td>\n",
       "      <td>287.162923</td>\n",
       "      <td>238.837177</td>\n",
       "      <td>45.594328</td>\n",
       "      <td>5.718922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6463</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>ZWE</td>\n",
       "      <td>2013</td>\n",
       "      <td>143.850145</td>\n",
       "      <td>113.456097</td>\n",
       "      <td>27.589603</td>\n",
       "      <td>4.426291</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6464</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>ZWE</td>\n",
       "      <td>2014</td>\n",
       "      <td>138.200536</td>\n",
       "      <td>108.703566</td>\n",
       "      <td>26.760618</td>\n",
       "      <td>4.296971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6465</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>ZWE</td>\n",
       "      <td>2015</td>\n",
       "      <td>132.752553</td>\n",
       "      <td>104.340506</td>\n",
       "      <td>25.715415</td>\n",
       "      <td>4.200907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6466</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>ZWE</td>\n",
       "      <td>2016</td>\n",
       "      <td>128.692138</td>\n",
       "      <td>100.392287</td>\n",
       "      <td>25.643570</td>\n",
       "      <td>4.117173</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6467</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>ZWE</td>\n",
       "      <td>2017</td>\n",
       "      <td>125.028843</td>\n",
       "      <td>96.235996</td>\n",
       "      <td>26.166182</td>\n",
       "      <td>4.052495</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6468 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           Entity Code  Year  Air pollution (total) (deaths per 100,000)  \\\n",
       "0     Afghanistan  AFG  1990                                  299.477309   \n",
       "1     Afghanistan  AFG  1991                                  291.277967   \n",
       "2     Afghanistan  AFG  1992                                  278.963056   \n",
       "3     Afghanistan  AFG  1993                                  278.790815   \n",
       "4     Afghanistan  AFG  1994                                  287.162923   \n",
       "...           ...  ...   ...                                         ...   \n",
       "6463     Zimbabwe  ZWE  2013                                  143.850145   \n",
       "6464     Zimbabwe  ZWE  2014                                  138.200536   \n",
       "6465     Zimbabwe  ZWE  2015                                  132.752553   \n",
       "6466     Zimbabwe  ZWE  2016                                  128.692138   \n",
       "6467     Zimbabwe  ZWE  2017                                  125.028843   \n",
       "\n",
       "      Indoor air pollution (deaths per 100,000)  \\\n",
       "0                                    250.362910   \n",
       "1                                    242.575125   \n",
       "2                                    232.043878   \n",
       "3                                    231.648134   \n",
       "4                                    238.837177   \n",
       "...                                         ...   \n",
       "6463                                 113.456097   \n",
       "6464                                 108.703566   \n",
       "6465                                 104.340506   \n",
       "6466                                 100.392287   \n",
       "6467                                  96.235996   \n",
       "\n",
       "      Outdoor particulate matter (deaths per 100,000)  \\\n",
       "0                                           46.446589   \n",
       "1                                           46.033841   \n",
       "2                                           44.243766   \n",
       "3                                           44.440148   \n",
       "4                                           45.594328   \n",
       "...                                               ...   \n",
       "6463                                        27.589603   \n",
       "6464                                        26.760618   \n",
       "6465                                        25.715415   \n",
       "6466                                        25.643570   \n",
       "6467                                        26.166182   \n",
       "\n",
       "      Outdoor ozone pollution (deaths per 100,000)  \n",
       "0                                         5.616442  \n",
       "1                                         5.603960  \n",
       "2                                         5.611822  \n",
       "3                                         5.655266  \n",
       "4                                         5.718922  \n",
       "...                                            ...  \n",
       "6463                                      4.426291  \n",
       "6464                                      4.296971  \n",
       "6465                                      4.200907  \n",
       "6466                                      4.117173  \n",
       "6467                                      4.052495  \n",
       "\n",
       "[6468 rows x 7 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"death_ratesfrom_airpollution.csv.xls\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c78aded1",
   "metadata": {},
   "outputs": [],
   "source": [
    "independent_vars = df[[\"Air pollution (total) (deaths per 100,000)\", \n",
    "                         \"Indoor air pollution (deaths per 100,000)\", \n",
    "                         \"Outdoor particulate matter (deaths per 100,000)\", \n",
    "                         \"Outdoor ozone pollution (deaths per 100,000)\"]]\n",
    "\n",
    "# Dependent variable\n",
    "dependent_var = df[\"Indoor air pollution (deaths per 100,000)\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "750f2ba1",
   "metadata": {},
   "outputs": [],
   "source": [
    "independent_vars = sm.add_constant(independent_vars)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "12670237",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                        OLS Regression Results                                       \n",
      "=====================================================================================================\n",
      "Dep. Variable:     Indoor air pollution (deaths per 100,000)   R-squared:                       1.000\n",
      "Model:                                                   OLS   Adj. R-squared:                  1.000\n",
      "Method:                                        Least Squares   F-statistic:                 6.242e+31\n",
      "Date:                                       Thu, 11 Jan 2024   Prob (F-statistic):               0.00\n",
      "Time:                                               18:53:11   Log-Likelihood:             1.7735e+05\n",
      "No. Observations:                                       6468   AIC:                        -3.547e+05\n",
      "Df Residuals:                                           6463   BIC:                        -3.547e+05\n",
      "Df Model:                                                  4                                         \n",
      "Covariance Type:                                   nonrobust                                         \n",
      "===================================================================================================================\n",
      "                                                      coef    std err          t      P>|t|      [0.025      0.975]\n",
      "-------------------------------------------------------------------------------------------------------------------\n",
      "const                                            2.442e-14    1.1e-14      2.213      0.027    2.79e-15    4.61e-14\n",
      "Air pollution (total) (deaths per 100,000)       1.477e-14   1.28e-14      1.157      0.247   -1.03e-14    3.98e-14\n",
      "Indoor air pollution (deaths per 100,000)           1.0000   1.27e-14   7.86e+13      0.000       1.000       1.000\n",
      "Outdoor particulate matter (deaths per 100,000) -6.883e-15   1.28e-14     -0.539      0.590   -3.19e-14    1.82e-14\n",
      "Outdoor ozone pollution (deaths per 100,000)    -5.551e-15   7.54e-15     -0.736      0.462   -2.03e-14    9.24e-15\n",
      "==============================================================================\n",
      "Omnibus:                      122.623   Durbin-Watson:                   0.039\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              210.819\n",
      "Skew:                          -0.156   Prob(JB):                     1.66e-46\n",
      "Kurtosis:                       3.828   Cond. No.                         854.\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n"
     ]
    }
   ],
   "source": [
    "model = sm.OLS(dependent_var, independent_vars).fit()\n",
    "print(model.summary())\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "722e29e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['Indoor air pollution (deaths per 100,000)', 'Outdoor particulate matter (deaths per 100,000)']]\n",
    "y = df['Air pollution (total) (deaths per 100,000)']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "90594b50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5050)\n",
    "\n",
    "# Create and train a Linear Regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69b5bbf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "coefficients = model.coef_\n",
    "intercept = model.intercept_\n",
    "\n",
    "print(\"Coefficients:\", coefficients)\n",
    "print(\"Intercept:\", intercept)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11e290a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model's performance\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "196866f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model using Mean Squared Error\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error: {mse}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "146ec391",
   "metadata": {},
   "source": [
    "# Random Forest Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a035f7b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a82493f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "model= RandomForestRegressor(n_estimators=56, random_state=30)\n",
    "model.fit(X_train, y_train)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6101a41d",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_estimators = 100  # Number of trees in the forest, adjust as needed\n",
    "random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=20)\n",
    "\n",
    "random_forest.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d8c32f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_importances = random_forest.feature_importances_\n",
    "print(\"Feature Importances:\", feature_importances)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13d9a08b",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = random_forest.predict(X_test)\n",
    "\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(\"Mean Squared Error:\", mse)\n",
    "print(\"R-squared:\", r2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07d66ee8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3b02913",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "c30f7088",
   "metadata": {},
   "source": [
    "# Gradient Boosting Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ec3a817",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c4c54df",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09194d4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model using Mean Squared Error\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error: {mse}\")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "666bae07",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "# Support Vector Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c10b7da",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "772d3bc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fa788765",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.iloc[:,0:-1]\n",
    "y=df.iloc[:,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b87c53a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6468, 7)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1acb8fb7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "shape of x train (5174, 6)\n",
      "shape of y train (5174,)\n",
      "Shape of x test (1294, 6)\n",
      "shape of y tets (1294,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,\n",
    "                                               random_state=2020)\n",
    "\n",
    "print(\"shape of x train\",x_train.shape)\n",
    "print(\"shape of y train\",y_train.shape)\n",
    "print(\"Shape of x test\",x_test.shape)\n",
    "print(\"shape of y tets\",y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0ba44fc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "bfbf06b0",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SVR' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[26], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m epsilon \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0.1\u001b[39m  \u001b[38;5;66;03m# Epsilon parameter for SVR, adjust as needed\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m svr\u001b[38;5;241m=\u001b[39m \u001b[43mSVR\u001b[49m(epsilon\u001b[38;5;241m=\u001b[39mepsilon)\n\u001b[0;32m      4\u001b[0m svr\u001b[38;5;241m.\u001b[39mfit(x_train_scaled, y_train)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SVR' is not defined"
     ]
    }
   ],
   "source": [
    "epsilon = 0.1  # Epsilon parameter for SVR, adjust as needed\n",
    "svr= SVR(epsilon=epsilon)\n",
    "\n",
    "svr.fit(x_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "446321d9",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'svr' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[12], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m y_pred \u001b[38;5;241m=\u001b[39m \u001b[43msvr\u001b[49m\u001b[38;5;241m.\u001b[39mpredict(X_test_scaled)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m# Calculate coefficient of determination (R-squared)\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmetrics\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m r2_score\n",
      "\u001b[1;31mNameError\u001b[0m: name 'svr' is not defined"
     ]
    }
   ],
   "source": [
    "y_pred = svr.predict(X_test_scaled)\n",
    "\n",
    "# Calculate coefficient of determination (R-squared)\n",
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "print(\"R-squared:\", r2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "af0f3072",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "shapes (1294,2) and (5,) not aligned: 2 (dim 1) != 5 (dim 0)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[13], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Make predictions on the test set\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m y_pred \u001b[38;5;241m=\u001b[39m \u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mpredict\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_test\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Evaluate the model using Mean Squared Error\u001b[39;00m\n\u001b[0;32m      5\u001b[0m mse \u001b[38;5;241m=\u001b[39m mean_squared_error(y_test, y_pred)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda35\\lib\\site-packages\\statsmodels\\base\\model.py:1159\u001b[0m, in \u001b[0;36mResults.predict\u001b[1;34m(self, exog, transform, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1156\u001b[0m         exog \u001b[38;5;241m=\u001b[39m exog[:, \u001b[38;5;28;01mNone\u001b[39;00m]\n\u001b[0;32m   1157\u001b[0m     exog \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39matleast_2d(exog)  \u001b[38;5;66;03m# needed in count model shape[1]\u001b[39;00m\n\u001b[1;32m-> 1159\u001b[0m predict_results \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmodel\u001b[38;5;241m.\u001b[39mpredict(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mparams, exog, \u001b[38;5;241m*\u001b[39margs,\n\u001b[0;32m   1160\u001b[0m                                      \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1162\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m exog_index \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mhasattr\u001b[39m(predict_results,\n\u001b[0;32m   1163\u001b[0m                                           \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpredicted_values\u001b[39m\u001b[38;5;124m'\u001b[39m):\n\u001b[0;32m   1164\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m predict_results\u001b[38;5;241m.\u001b[39mndim \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda35\\lib\\site-packages\\statsmodels\\regression\\linear_model.py:397\u001b[0m, in \u001b[0;36mRegressionModel.predict\u001b[1;34m(self, params, exog)\u001b[0m\n\u001b[0;32m    394\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m exog \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    395\u001b[0m     exog \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mexog\n\u001b[1;32m--> 397\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdot\u001b[49m\u001b[43m(\u001b[49m\u001b[43mexog\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mparams\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m<__array_function__ internals>:180\u001b[0m, in \u001b[0;36mdot\u001b[1;34m(*args, **kwargs)\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: shapes (1294,2) and (5,) not aligned: 2 (dim 1) != 5 (dim 0)"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model using Mean Squared Error\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error: {mse}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5052804e",
   "metadata": {},
   "source": [
    "# Ridge And Lasso Rigression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62433109",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import Ridge\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ac91d76",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import Lasso "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a843cc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47799c14",
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso_model = Lasso(alpha=0.08)  # You can adjust the regularization strength (alpha)\n",
    "lasso_model.fit(X_train, y_train)\n",
    "scaler = StandardScaler()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20e28af8",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02b4ef17",
   "metadata": {},
   "outputs": [],
   "source": [
    "alpha = 1.0  # Regularization strength, you can adjust this parameter\n",
    "ridge = Ridge(alpha=alpha)\n",
    "\n",
    "ridge.fit(x_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b23a0745",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'ridge' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[14], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m coefficients \u001b[38;5;241m=\u001b[39m \u001b[43mridge\u001b[49m\u001b[38;5;241m.\u001b[39mcoef_\n\u001b[0;32m      2\u001b[0m intercept \u001b[38;5;241m=\u001b[39m ridge\u001b[38;5;241m.\u001b[39mintercept_\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCoefficients:\u001b[39m\u001b[38;5;124m\"\u001b[39m, coefficients)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'ridge' is not defined"
     ]
    }
   ],
   "source": [
    "coefficients = ridge.coef_\n",
    "intercept = ridge.intercept_\n",
    "\n",
    "print(\"Coefficients:\", coefficients)\n",
    "print(\"Intercept:\", intercept)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ddb3b5d6",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'lasso_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[15], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Make predictions on the test set\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m y_pred_lasso \u001b[38;5;241m=\u001b[39m \u001b[43mlasso_model\u001b[49m\u001b[38;5;241m.\u001b[39mpredict(X_test)\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Evaluate the Lasso model using Mean Squared Error\u001b[39;00m\n\u001b[0;32m      5\u001b[0m mse_lasso \u001b[38;5;241m=\u001b[39m mean_squared_error(y_test, y_pred_lasso)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'lasso_model' is not defined"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set\n",
    "y_pred_lasso = lasso_model.predict(X_test)\n",
    "\n",
    "# Evaluate the Lasso model using Mean Squared Error\n",
    "mse_lasso = mean_squared_error(y_test, y_pred_lasso)\n",
    "print(f\"Lasso Mean Squared Error: {mse_lasso}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "904dcc49",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Ridge' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[16], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Initialize and fit the Ridge Regression model\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m ridge_model\u001b[38;5;241m=\u001b[39m \u001b[43mRidge\u001b[49m(alpha\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.1\u001b[39m)  \u001b[38;5;66;03m# You can adjust the regularization strength (alpha)\u001b[39;00m\n\u001b[0;32m      3\u001b[0m ridge_model\u001b[38;5;241m.\u001b[39mfit(X_train, y_train)\n\u001b[0;32m      5\u001b[0m \u001b[38;5;66;03m# Make predictions on the test set\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'Ridge' is not defined"
     ]
    }
   ],
   "source": [
    "# Initialize and fit the Ridge Regression model\n",
    "ridge_model= Ridge(alpha=0.1)  # You can adjust the regularization strength (alpha)\n",
    "ridge_model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test set\n",
    "y_pred_ridge = ridge_model.predict(X_test)\n",
    "mse_ridge = mean_squared_error(y_test, y_pred_ridge)\n",
    "print(f\"Ridge Mean Squared Error: {mse_ridge}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "92bced92",
   "metadata": {},
   "source": [
    "# Elastic Net Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f56daf7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import ElasticNet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9def431f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>ElasticNet(alpha=0.01)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">ElasticNet</label><div class=\"sk-toggleable__content\"><pre>ElasticNet(alpha=0.01)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "ElasticNet(alpha=0.01)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "elastic_net_model = ElasticNet(alpha=0.01, l1_ratio=0.5)  # You can adjust alpha and l1_ratio\n",
    "elastic_net_model.fit(X_train, y_train)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2739bdb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Elastic Net Mean Squared Error: 18.893162067026484\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "y_pred_elastic_net = elastic_net_model.predict(X_test)\n",
    "\n",
    "\n",
    "# Evaluate the Elastic Net model using Mean Squared Error\n",
    "mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)\n",
    "print(f\"Elastic Net Mean Squared Error: {mse_elastic_net}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "58a48f9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6468, 7)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fb4dd569",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Entity</th>\n",
       "      <th>Code</th>\n",
       "      <th>Year</th>\n",
       "      <th>Air pollution (total) (deaths per 100,000)</th>\n",
       "      <th>Indoor air pollution (deaths per 100,000)</th>\n",
       "      <th>Outdoor particulate matter (deaths per 100,000)</th>\n",
       "      <th>Outdoor ozone pollution (deaths per 100,000)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6463</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6464</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6465</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6466</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6467</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6468 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Entity   Code   Year  Air pollution (total) (deaths per 100,000)  \\\n",
       "0      False  False  False                                       False   \n",
       "1      False  False  False                                       False   \n",
       "2      False  False  False                                       False   \n",
       "3      False  False  False                                       False   \n",
       "4      False  False  False                                       False   \n",
       "...      ...    ...    ...                                         ...   \n",
       "6463   False  False  False                                       False   \n",
       "6464   False  False  False                                       False   \n",
       "6465   False  False  False                                       False   \n",
       "6466   False  False  False                                       False   \n",
       "6467   False  False  False                                       False   \n",
       "\n",
       "      Indoor air pollution (deaths per 100,000)  \\\n",
       "0                                         False   \n",
       "1                                         False   \n",
       "2                                         False   \n",
       "3                                         False   \n",
       "4                                         False   \n",
       "...                                         ...   \n",
       "6463                                      False   \n",
       "6464                                      False   \n",
       "6465                                      False   \n",
       "6466                                      False   \n",
       "6467                                      False   \n",
       "\n",
       "      Outdoor particulate matter (deaths per 100,000)  \\\n",
       "0                                               False   \n",
       "1                                               False   \n",
       "2                                               False   \n",
       "3                                               False   \n",
       "4                                               False   \n",
       "...                                               ...   \n",
       "6463                                            False   \n",
       "6464                                            False   \n",
       "6465                                            False   \n",
       "6466                                            False   \n",
       "6467                                            False   \n",
       "\n",
       "      Outdoor ozone pollution (deaths per 100,000)  \n",
       "0                                            False  \n",
       "1                                            False  \n",
       "2                                            False  \n",
       "3                                            False  \n",
       "4                                            False  \n",
       "...                                            ...  \n",
       "6463                                         False  \n",
       "6464                                         False  \n",
       "6465                                         False  \n",
       "6466                                         False  \n",
       "6467                                         False  \n",
       "\n",
       "[6468 rows x 7 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "7725100c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Entity  Code  Year  Air pollution (total) (deaths per 100,000)  \\\n",
      "0       0     0  1990                                  299.477309   \n",
      "1       0     0  1991                                  291.277967   \n",
      "2       0     0  1992                                  278.963056   \n",
      "3       0     0  1993                                  278.790815   \n",
      "4       0     0  1994                                  287.162923   \n",
      "\n",
      "   Indoor air pollution (deaths per 100,000)  \\\n",
      "0                                 250.362910   \n",
      "1                                 242.575125   \n",
      "2                                 232.043878   \n",
      "3                                 231.648134   \n",
      "4                                 238.837177   \n",
      "\n",
      "   Outdoor particulate matter (deaths per 100,000)  \\\n",
      "0                                        46.446589   \n",
      "1                                        46.033841   \n",
      "2                                        44.243766   \n",
      "3                                        44.440148   \n",
      "4                                        45.594328   \n",
      "\n",
      "   Outdoor ozone pollution (deaths per 100,000)  \n",
      "0                                      5.616442  \n",
      "1                                      5.603960  \n",
      "2                                      5.611822  \n",
      "3                                      5.655266  \n",
      "4                                      5.718922  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "\n",
    "# Initialize LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "\n",
    "# Encode Entity and Code columns\n",
    "df[\"Entity\"] = label_encoder.fit_transform(df[\"Entity\"])\n",
    "df[\"Code\"] = label_encoder.fit_transform(df[\"Code\"])\n",
    "\n",
    "# Now you can use the encoded data for analysis or machine learning\n",
    "print(df.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "720dc3f7",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'logreg' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[23], line 7\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlinear_model\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m LogisticRegression\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Extract coefficients\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m coefficients \u001b[38;5;241m=\u001b[39m \u001b[43mlogreg\u001b[49m\u001b[38;5;241m.\u001b[39mcoef_\n\u001b[0;32m      8\u001b[0m intercept \u001b[38;5;241m=\u001b[39m logreg\u001b[38;5;241m.\u001b[39mintercept_\n\u001b[0;32m     10\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCoefficients:\u001b[39m\u001b[38;5;124m\"\u001b[39m, coefficients)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'logreg' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# Extract coefficients\n",
    "coefficients = logreg.coef_\n",
    "intercept = logreg.intercept_\n",
    "\n",
    "print(\"Coefficients:\", coefficients)\n",
    "print(\"Intercept:\", intercept)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "916814c5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c73361c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9576277d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16c3f03d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fe5f974",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "51567253",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Entity                                             0\n",
       "Code                                               0\n",
       "Year                                               0\n",
       "Air pollution (total) (deaths per 100,000)         0\n",
       "Indoor air pollution (deaths per 100,000)          0\n",
       "Outdoor particulate matter (deaths per 100,000)    0\n",
       "Outdoor ozone pollution (deaths per 100,000)       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2118d338",
   "metadata": {},
   "source": [
    "Linear Regression:\n",
    "\n",
    "Use when you assume a linear relationship between independent and dependent variables.\n",
    "Suitable for simpler models where assumptions hold.\n",
    "Can serve as a baseline for more complex models.\n",
    "Random Forest Regression:\n",
    "\n",
    "Effective for capturing non-linear relationships and handling complex interactions.\n",
    "Handles well both numerical and categorical features.\n",
    "Robust against overfitting due to ensemble nature.\n",
    "Gradient Boosting Regression:\n",
    "\n",
    "Powerful for capturing complex relationships in data.\n",
    "Can handle both numerical and categorical features.\n",
    "Often provides better predictive accuracy but requires careful hyperparameter tuning.\n",
    "Support Vector Regression (SVR):\n",
    "\n",
    "Useful for non-linear regression tasks.\n",
    "Effective when there is limited training data and noise.\n",
    "Can handle both linear and non-linear relationships using different kernel functions.\n",
    "Lasso and Ridge Regression:\n",
    "\n",
    "Lasso (L1 regularization) is useful for feature selection and handling multicollinearity.\n",
    "Ridge (L2 regularization) helps prevent overfitting and stabilizes coefficient estimates.\n",
    "Both are effective when you have a high-dimensional dataset with potential collinearity.\n",
    "Elastic Net Regression:\n",
    "\n",
    "Combines benefits of Lasso and Ridge by balancing L1 and L2 regularization.\n",
    "Useful when dealing with high-dimensional data and collinearity.\n",
    "Offers flexibility in handling different types of features.\n",
    "Conclusion:\n",
    "The best algorithm depends on the complexity of relationships in your data, the presence of multicollinearity, the size of your dataset, and your goals. It's a good practice to try multiple algorithms and evaluate their performance using appropriate metrics (e.g., Mean Squared Error) and cross-validation. Additionally, hyperparameter tuning can significantly impact the performance of these algorithms.\n",
    "\n",
    "Remember that while selecting the algorithm is important, other steps such as data preprocessing, feature engineering, and proper evaluation also play a crucial role in achieving accurate and reliable predictions. Always tailor your choice to the specifics of your data and analysis goals.\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "fdfffe94",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "\"['Air pollution (total)', 'deaths per 100,000', 'Indoor air pollution deaths per 100,000', 'Outdoor particulate matter deaths per 100,000', 'Outdoor ozone pollution deaths per 100,000'] not in index\"",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[31], line 5\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlinear_model\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m LinearRegression\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Assuming your data is stored in a pandas DataFrame called 'data'\u001b[39;00m\n\u001b[1;32m----> 5\u001b[0m features \u001b[38;5;241m=\u001b[39m \u001b[43mdf\u001b[49m\u001b[43m[\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mYear\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mAir pollution (total)\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mdeaths per 100,000\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m      6\u001b[0m \u001b[43m               \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mIndoor air pollution deaths per 100,000\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m      7\u001b[0m \u001b[43m               \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mOutdoor particulate matter deaths per 100,000\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m      8\u001b[0m \u001b[43m               \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mOutdoor ozone pollution deaths per 100,000\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m]\u001b[49m  \n\u001b[0;32m      9\u001b[0m                \u001b[38;5;66;03m# Choose appropriate features\u001b[39;00m\n\u001b[0;32m     10\u001b[0m target \u001b[38;5;241m=\u001b[39m df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mValue 1\u001b[39m\u001b[38;5;124m'\u001b[39m]  \u001b[38;5;66;03m# Choose the target variable\u001b[39;00m\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda35\\lib\\site-packages\\pandas\\core\\frame.py:3813\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3811\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m is_iterator(key):\n\u001b[0;32m   3812\u001b[0m         key \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(key)\n\u001b[1;32m-> 3813\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcolumns\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_get_indexer_strict\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mcolumns\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m[\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m   3815\u001b[0m \u001b[38;5;66;03m# take() does not accept boolean indexers\u001b[39;00m\n\u001b[0;32m   3816\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mgetattr\u001b[39m(indexer, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdtype\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m) \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mbool\u001b[39m:\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda35\\lib\\site-packages\\pandas\\core\\indexes\\base.py:6070\u001b[0m, in \u001b[0;36mIndex._get_indexer_strict\u001b[1;34m(self, key, axis_name)\u001b[0m\n\u001b[0;32m   6067\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   6068\u001b[0m     keyarr, indexer, new_indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_reindex_non_unique(keyarr)\n\u001b[1;32m-> 6070\u001b[0m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_raise_if_missing\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkeyarr\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mindexer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43maxis_name\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   6072\u001b[0m keyarr \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtake(indexer)\n\u001b[0;32m   6073\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Index):\n\u001b[0;32m   6074\u001b[0m     \u001b[38;5;66;03m# GH 42790 - Preserve name from an Index\u001b[39;00m\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda35\\lib\\site-packages\\pandas\\core\\indexes\\base.py:6133\u001b[0m, in \u001b[0;36mIndex._raise_if_missing\u001b[1;34m(self, key, indexer, axis_name)\u001b[0m\n\u001b[0;32m   6130\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNone of [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mkey\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m] are in the [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00maxis_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m]\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m   6132\u001b[0m not_found \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(ensure_index(key)[missing_mask\u001b[38;5;241m.\u001b[39mnonzero()[\u001b[38;5;241m0\u001b[39m]]\u001b[38;5;241m.\u001b[39munique())\n\u001b[1;32m-> 6133\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mnot_found\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m not in index\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mKeyError\u001b[0m: \"['Air pollution (total)', 'deaths per 100,000', 'Indoor air pollution deaths per 100,000', 'Outdoor particulate matter deaths per 100,000', 'Outdoor ozone pollution deaths per 100,000'] not in index\""
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Assuming your data is stored in a pandas DataFrame called 'data'\n",
    "features = df[['Year', 'Air pollution (total)','deaths per 100,000',\n",
    "               'Indoor air pollution deaths per 100,000',\n",
    "               'Outdoor particulate matter deaths per 100,000',\n",
    "               'Outdoor ozone pollution deaths per 100,000']]  \n",
    "               # Choose appropriate features\n",
    "target = df['Value 1']  # Choose the target variable\n",
    "\n",
    "# Splitting the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)\n",
    "\n",
    "# Creating and training the Linear Regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Extracting coefficients and intercept\n",
    "coefficients = model.coef_\n",
    "intercept = model.intercept_\n",
    "\n",
    "print(\"Coefficients:\", coefficients)\n",
    "print(\"Intercept:\", intercept)\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30e656a3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8bea93ff",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef16143f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "384c6e93",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ae336cf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0bb32011",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9f8d154",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
