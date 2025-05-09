from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the trained model
model = load_model("earthquake10_model.h5")

# HTML Templates for the Frontend
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Earthquake Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --glass-bg: rgba(255, 255, 255, 0.15);
      --text-light: #fff;
      --text-dark: #222;
    }

    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #3a4346, #4b74a5);
      transition: background 0.5s ease;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      color: var(--text-light);
    }

    body.dark {
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      color: var(--text-light);
    }

    .glass-card {
      background: var(--glass-bg);
      box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
      backdrop-filter: blur(12px);
      border-radius: 20px;
      padding: 40px;
      width: 90%;
      max-width: 450px;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.18);
    }

    h1 {
      margin-bottom: 25px;
      font-size: 28px;
    }

    input[type="file"] {
      width: 100%;
      padding: 12px;
      border-radius: 12px;
      background-color: rgba(255, 255, 255, 0.2);
      border: 1px dashed #ffffffa8;
      color: #fff;
      margin-bottom: 20px;
    }

    input[type="file"]::file-selector-button {
      background: #ffffff10;
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 10px;
      cursor: pointer;
    }

    button {
      padding: 12px 24px;
      background: linear-gradient(135deg, #6a11cb, #2575fc);
      border: none;
      border-radius: 12px;
      color: white;
      font-size: 16px;
      font-weight: 500;
      cursor: pointer;
      transition: transform 0.3s ease;
    }

    button:hover {
      transform: scale(1.05);
    }

    .loader {
      display: none;
      margin-top: 20px;
      border: 5px solid rgba(255, 255, 255, 0.2);
      border-top: 5px solid #fff;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin-left: auto;
      margin-right: auto;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .toggle {
      margin-top: 25px;
      cursor: pointer;
      font-size: 14px;
      color: #eee;
    }

    footer {
      margin-top: 30px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="glass-card">
    <h1>🌍 Earthquake Prediction</h1>
    <form id="upload-form" action="/upload" method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required />
      <button type="submit">Upload and Predict</button>
      <div class="loader" id="loader"></div>
    </form>
    <div class="toggle" onclick="toggleMode()">🌗 Toggle Light Mode/Dark Mode</div>
    <footer>Developed by <strong>Shreyan</strong></footer>
  </div>

  <script>
    document.getElementById('upload-form').addEventListener('submit', function () {
      document.getElementById('loader').style.display = 'block';
    });

    function toggleMode() {
      document.body.classList.toggle('dark');
    }
  </script>
</body>
</html>
'''

RESULT_HTML = '''
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Prediction Result</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-light: linear-gradient(135deg, #85a4ae, #7f9bbd);
      --bg-dark: linear-gradient(135deg, #24427a, #2a5298);
      --text-light: #111;
      --text-dark: #fff;
    }

    html[data-theme='light'] {
      --bg: var(--bg-light);
      --text: var(--text-light);
    }

    html[data-theme='dark'] {
      --bg: var(--bg-dark);
      --text: var(--text-dark);
    }

    body {
      font-family: 'Poppins', sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      transition: background 0.4s ease, color 0.4s ease;
    }

    .glass-card {
      background: rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border-radius: 20px;
      padding: 40px;
      width: 90%;
      max-width: 700px;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    h1 {
      font-size: 26px;
    }

    .summary {
      background: rgba(255, 255, 255, 0.15);
      padding: 16px;
      margin: 20px 0;
      border-radius: 12px;
    }

    .summary h3 {
      margin-bottom: 10px;
    }

    canvas {
      margin-top: 20px;
      width: 100% !important;
      max-height: 300px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.1);
    }

    a {
      display: inline-block;
      margin-top: 25px;
      background: linear-gradient(135deg, #00c6ff, #0072ff);
      color: white;
      padding: 12px 24px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 500;
      transition: transform 0.3s ease;
    }

    a:hover {
      transform: scale(1.05);
    }

    .toggle-btn {
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(255,255,255,0.1);
      border: none;
      color: var(--text);
      padding: 10px 16px;
      border-radius: 10px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <button class="toggle-btn" onclick="toggleTheme()">🌓 Toggle Theme</button>
  <div class="glass-card">
    <h1>📈 Prediction Result</h1>
    <p>Predicted time to failure: <strong>{{ prediction }}</strong> seconds</p>

    <div class="summary">
      <h3>📊 Data Summary</h3>
      <p>Min Value: {{ min_val }}</p>
      <p>Max Value: {{ max_val }}</p>
      <p>Mean Value: {{ mean_val }}</p>
    </div>

    <canvas id="waveform"></canvas>
    <a href="/">← Upload Another File</a>
  </div>

  <script>
    function toggleTheme() {
      const html = document.documentElement;
      html.setAttribute('data-theme', html.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
    }

    const waveformData = {{ waveform | tojson }};
    const ctx = document.getElementById('waveform').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: waveformData.map((_, i) => i),
        datasets: [{
          label: 'Acoustic Data',
          data: waveformData,
          borderColor: '#00fff7',
          backgroundColor: 'rgba(0,255,247,0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { color: '#fff' },
            title: { display: true, text: 'Index', color: '#fff' }
          },
          y: {
            ticks: { color: '#fff' },
            title: { display: true, text: 'Acoustic Value', color: '#fff' }
          }
        }
      }
    });
  </script>
</body>
</html>
'''

# Routes for the Flask app
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            # Save file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            chunk_size = 20000
            reader = pd.read_csv(tmp_path, chunksize=chunk_size)

            X_test_new = []
            y_test_new = []

            # Read and process the chunks from CSV
            for chunk in reader:
                if len(chunk) < chunk_size:
                    continue

                # Normalize the acoustic data
                x = chunk['acoustic_data'].values.astype(np.float32)
                x = (x - x.mean()) / (x.std() + 1e-6)
                X_test_new.append(x.reshape(chunk_size, 1))

                # Collect the last time_to_failure value
                if 'time_to_failure' in chunk.columns:
                    y = chunk['time_to_failure'].values[-1]
                    y_test_new.append(y)

            # Convert lists to numpy arrays
            X_test_new = np.array(X_test_new, dtype=np.float32)
            y_test_new = np.array(y_test_new, dtype=np.float32) if y_test_new else None 

            # Make predictions
            y_pred = model.predict(X_test_new)
            predicted_value = float(y_pred[0][0])

            # Send data for visualization
            return render_template_string(
                RESULT_HTML,
                prediction=round(predicted_value, 6),
                waveform=x.tolist()  # Acoustic data for the graph
            )

        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        return jsonify({"error": "Invalid file type. Only .csv files are allowed."}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
