from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

# StandardScaler for feature scaling
standard_scaler = StandardScaler()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # Retrieve form data
        prb_usage = int(request.form['PRB_usage'])
        mcs = int(request.form['MCS'])
        rrc_conn_ue = int(request.form['RRC_conn_UE'])
        
        # Preprocess the input features
        features = np.array([[prb_usage, rrc_conn_ue, mcs]])
        features_scaled = standard_scaler.fit_transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        throughput = round(prediction[0], 2)
        
        # Render the result template with the prediction
        return render_template('result.html', throughput=throughput)
    
if __name__ == "__main__":
    app.run(debug=True)
