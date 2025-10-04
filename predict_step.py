#Ahmed Negm | 501101640 
# AER 850 Machine Learning 

import joblib  # Import joblib to load the model
import pandas as pd  

# Load the saved SVC model
loaded_model = joblib.load('best_svc_model.joblib')

# Coordinates to predict the corresponding maintenance step
coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# converting the coordinates to a pandas dataframe
df_coordinates = pd.DataFrame(coordinates, columns=['X', 'Y', 'Z'])

# predicting the class labels for the coordinates
predictions = loaded_model.predict(df_coordinates)

# printing the predictions fpr our model.
print("Predictions for the given coordinates:")
for coord, pred in zip(coordinates, predictions):
    print(f"Coordinates: {coord} -> Predicted Maintenance Step: {pred}")
