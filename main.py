import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('Students Social Media Addiction.csv')

# Display first rows & info
print("Dataset preview:")
print(df.head())
print("\nData info:")
print(df.info())

# Define addiction levels based on 'Addicted_Score'
def addiction_label(score):
    if score <= 3:
        return 'Low'
    elif score <= 7:
        return 'Moderate'
    else:
        return 'High'

df['Addiction_Level'] = df['Addicted_Score'].apply(addiction_label)

print("\nAddiction Level distribution:")
print(df['Addiction_Level'].value_counts())

# Plotting addiction level counts (optional)
plt.figure(figsize=(6,4))
sns.countplot(x='Addiction_Level', data=df, palette='pastel')
plt.title('Addiction Level Counts')
plt.show()

# Prepare features and target
predictors = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
X = df[predictors]
y = df['Addiction_Level']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
