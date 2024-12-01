import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("loan.csv")

# Remove missing values
df.dropna(inplace=True)
df.drop(columns=['Loan_ID'], inplace=True)


# Check for missing values
print("Missing values in dataset:\n", df.isnull().sum())

# Encode categorical columns
encode = LabelEncoder()
df["Loan_Status"] = encode.fit_transform(df["Loan_Status"])
df["Gender"] = encode.fit_transform(df["Gender"])
df["Married"] = encode.fit_transform(df["Married"])
df["Education"] = encode.fit_transform(df["Education"])
df["Self_Employed"] = encode.fit_transform(df["Self_Employed"])
df["Property_Area"] = encode.fit_transform(df["Property_Area"])
df["Dependents"] = encode.fit_transform(df["Dependents"])

# Standardize numerical columns
# scale = StandardScaler()
# df[["ApplicantIncome"]] = scale.fit_transform(df[["ApplicantIncome"]])
# df[["CoapplicantIncome"]] = scale.fit_transform(df[["CoapplicantIncome"]])
# df[["LoanAmount"]] = scale.fit_transform(df[["LoanAmount"]])
# df[["Loan_Amount_Term"]] = scale.fit_transform(df[["Loan_Amount_Term"]])

# Display correlation matrix
cor = df.corr()
loan_status_corr = cor["Loan_Status"]
print("Correlation with Loan_Status:\n", loan_status_corr)

# Split the data into features (X) and target (y)
x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
y = df['Loan_Status']


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")
