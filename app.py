from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#load data
df = pd.read_csv("./customer_data/Customer_Churn_new.csv",header=0)

# define the feature and target viarables
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert categorical columns to numerical values using one-hot encoding
ohe = OneHotEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns
categoric_data = ohe.fit_transform(X[categorical_columns]).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names_out()

# Identify numerical features
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# normalizing the features
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
numeric_df = pd.DataFrame(X[numerical_columns], dtype = object)

#combine numeric and categorix
X_final = pd.concat([numeric_df, categoric_df], axis = 1)

#train model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_final, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)

    data_categoric = np.array([data["Complaint"], data["Age_Group"], data["Tariff_Plan"], data["Status"]])
    data_categoric = np.reshape(data_categoric, (1, -1))
    data_categoric = ohe.transform(data_categoric).toarray()

    data_numerical = np.array([data["Call_Failure"], data["Tenure"], data["Charge_Amount"], data["Seconds_of_Use"], data["Frequency_of_use"], 
                            data["Frequency_of_SMS"], data["Distinct_Called_Numbers"],data["Age"],data["Customer_Value"]])
    data_numerical = np.reshape(data_numerical, (1, -1))
    data_numerical = np.array(scaler.transform(data_numerical))
 
    data_final = np.column_stack((data_numerical, data_categoric))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    predicted_class_label = 'No' if prediction[0] == 0 else 'Yes'
    return Response(json.dumps(predicted_class_label))

if __name__ == '__main__':
    app.run(port=3000, debug=True)