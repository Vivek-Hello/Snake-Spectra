# snakespectra/views.py

from django.shortcuts import render
import pandas as pd
import os
from django.http import HttpResponse
import joblib

# Load the model and label encoders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'snake_classifier.pkl')
color_encoder_path = os.path.join(BASE_DIR, 'label_encoder_color.pkl')
scales_encoder_path = os.path.join(BASE_DIR, 'label_encoder_scales.pkl')

classifier = joblib.load(model_path)
label_encoder_color = joblib.load(color_encoder_path)
label_encoder_scales = joblib.load(scales_encoder_path)

# Load the cleaned dataset for reference

df = pd.read_csv(r'cleaned_snakes_dataset.csv')

def find_snake_by_attributes(request):
    if request.method == 'POST':
        color = request.POST.get('color', '').strip().capitalize().title()
        scales = request.POST.get('scale', '').strip().capitalize()

        # Encode the input features
        '''
        try:
            color_encoded = label_encoder_color.transform([color])[0]
            scales_encoded = label_encoder_scales.transform([scales])[0]
        except ValueError:
            # Handle the case where the input color or scale is not found in the label encoder
            return render(request, 'error.html', {'message': 'Invalid input: color or scale not recognized'})

        # Predict venomous status
        prediction = classifier.predict([[color_encoded, scales_encoded]])
        venomous = 'Yes' if prediction[0] == 1 else 'No'
        '''
        df['Venomous'] = df['Venomous'].map({'Yes': 1, 'No': 0})

        # Find the matching snakes in the dataset
        result = df[(df['Color'] == color) & (df['Scales'] == scales)]

        snakes = []
        if not result.empty:
            for index, row in result.iterrows():
                name = row['Name']
                venomous=row['Venomous']
                snakes.append({'name': name, 'venomous': venomous})
            context = {'snakes': snakes}
            return render(request, 'result.html', context)
        else:
            return render(request, 'no_snake_found.html')
    else:
        return render(request, 'front.html')
