from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

print(iris.data.shape)
print(iris.target.shape)
print(iris.data[0])
print(iris.target[0])

# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['result'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# print(df.head(10))

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_jobs=2, random_state=0)
# n_job : تعداد کار همزمان را مشخص می کند

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(score)

y_pred = model.predict(x_test)
print(y_pred)
print(model.predict_proba(x_test))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predict label')
plt.title(f'Score : {score}', size=15)
plt.show()

print(iris.target_names)