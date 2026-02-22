import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from google.colab import files

# Upload file
uploaded = files.upload()

# Load dataset
data = pd.read_csv("news_big.csv")

print(data.head())
# Load dataset
data = pd.read_csv("news_big.csv")

# Input and Output
X = data['text']
y = data['label']

# Split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = RandomForestClassifier()

model.fit(X_train,y_train)

# Prediction
pred = model.predict(X_test)



# Accuracy
print("Accuracy:",accuracy_score(y_test,pred))


