import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
df = pd.read_csv('cleaned_snakes_dataset.csv')

# Encode categorical variables
label_encoder_color = LabelEncoder()
label_encoder_scales = LabelEncoder()

df['Color'] = label_encoder_color.fit_transform(df['Color'])
df['Scales'] = label_encoder_scales.fit_transform(df['Scales'])

# Split the data into features and target
X = df[['Color', 'Scales']]
y = df['Venomous']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Ensure the models directory exists
models_dir = os.path.join('snakespectra', 'models')
os.makedirs(models_dir, exist_ok=True)

# Save the model and label encoders
joblib.dump(classifier, os.path.join(models_dir, 'snake_classifier.pkl'))
joblib.dump(label_encoder_color, os.path.join(models_dir, 'label_encoder_color.pkl'))
joblib.dump(label_encoder_scales, os.path.join(models_dir, 'label_encoder_scales.pkl'))
