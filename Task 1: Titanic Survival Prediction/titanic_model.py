import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("Titanic_Dataset.csv")
print(df.head())

# Drop irrelevant columns
# Drop 'Ticket' if exists
if 'Ticket' in df.columns:
    df.drop("Ticket", axis=1, inplace=True)

# Drop 'Cabin' if exists
if 'Cabin' in df.columns:
    df.drop("Cabin", axis=1, inplace=True)


# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert Sex to numeric: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numeric: S=0, C=1, Q=2
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Confirm cleaning
print(df.isnull().sum())
print(df.head())

# Convert 'Survived' to numeric: Dead=0, Alive=1
df['Survived'] = df['Survived'].map({'Dead': 0, 'Alive': 1})

# Split into features (X) and target (y)
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
