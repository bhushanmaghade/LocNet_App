import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# --- CONFIG ---
LOOK_BACK = 20
IMG_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HELPER FUNCTIONS ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_occupancy_grid(sequence, img_size=32):
    img = np.zeros((img_size, img_size), dtype=np.float32)
    min_val = np.min(sequence, axis=0)
    max_val = np.max(sequence, axis=0)
    denom = max_val - min_val
    denom[denom == 0] = 1e-6 
    norm_seq = (sequence - min_val) / denom
    indices = (norm_seq * (img_size - 1)).astype(int)
    num_points = len(indices)
    for i in range(num_points):
        r, c = indices[i, 0], indices[i, 1]
        row_idx = img_size - 1 - r
        col_idx = c
        if 0 <= row_idx < img_size and 0 <= col_idx < img_size:
            intensity = (i + 1) / num_points 
            img[row_idx, col_idx] = intensity
    return img[np.newaxis, :, :]

# --- MODELS ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GPS_CNN(nn.Module):
    def __init__(self):
        super(GPS_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- HELPER: SAFE JSON ---
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(SafeJSONEncoder, self).default(obj)

app.json_encoder = SafeJSONEncoder

# --- ROUTES ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        df = pd.read_csv(file)
        
        # Identify columns
        cols = df.columns
        lat_col = next((c for c in cols if 'lat' in c.lower()), None)
        lon_col = next((c for c in cols if any(x in c.lower() for x in ['lon', 'lng'])), None)
        hdop_col = next((c for c in cols if 'hdop' in c.lower()), None)
        
        if not lat_col or not lon_col:
            return jsonify({"error": "Latitude/Longitude columns not found"}), 400
            
        # Clean Data
        df = df.rename(columns={lat_col: 'Lat', lon_col: 'Lon'})
        if hdop_col:
            df = df.rename(columns={hdop_col: 'HDOP'})
            df['HDOP'] = pd.to_numeric(df['HDOP'], errors='coerce')
        
        df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
        df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
        df = df.dropna(subset=['Lat', 'Lon'])
        
        # Save to DB (Optional / Fallback)
        conn = get_db_connection()
        upload_id = None
        
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO uploads (filename) VALUES (%s)", (file.filename,))
                upload_id = cursor.lastrowid
                
                # Bulk Insert Points
                points_data = []
                for _, row in df.iterrows():
                    hdop_val = row['HDOP'] if 'HDOP' in df.columns else None
                    if pd.isna(hdop_val): hdop_val = None
                    points_data.append((upload_id, row['Lat'], row['Lon'], hdop_val))
                    
                cursor.executemany("""
                    INSERT INTO gps_points (upload_id, lat, lon, hdop) 
                    VALUES (%s, %s, %s, %s)
                """, points_data)
                
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as db_err:
                print(f"DB Error: {db_err}")
                if conn: conn.close()
                # Continue without DB if insert fails
        else:
            print("Warning: No Database Connection. Using in-memory fallback.")
            
        # If DB failed or didn't exist, we send back ID=0 but cache data in simple file/memory?
        # For this demo, let's just assume we return success. 
        # In a real "no-db" mode, we'd serialize DF to temp file.
        # Let's use a temp csv approach if DB is missing to support the user's "update code" request.
        if not upload_id:
            upload_id = 99999 # Mock ID
            df.to_csv(f"temp_{upload_id}.csv", index=False)
        
        return jsonify({"message": "File uploaded successfully", "upload_id": upload_id, "rows": len(df)}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<int:upload_id>', methods=['GET'])
def predict(upload_id):
    try:
        df = pd.DataFrame()
        
        # Try Loading from DB
        conn = get_db_connection()
        if conn and upload_id != 99999:
            try:
                df = pd.read_sql(f"SELECT lat as Lat, lon as Lon, hdop as HDOP FROM gps_points WHERE upload_id = {upload_id} ORDER BY id ASC", conn)
                conn.close()
            except:
                if conn: conn.close()
        
        # Fallback: Temp CSV
        if df.empty or upload_id == 99999:
            if os.path.exists(f"temp_{upload_id}.csv"):
                df = pd.read_csv(f"temp_{upload_id}.csv")
            else:
                return jsonify({"error": "No data found (DB failed and no temp file)"}), 404
            
        # 1. Median
        lat_med = df['Lat'].median()
        lon_med = df['Lon'].median()
        
        # Handle NaN standard types
        if pd.isna(lat_med): lat_med = None
        if pd.isna(lon_med): lon_med = None
        
        # 2. Kalman Filter
        lat_kf, lon_kf = None, None
        try:
            if len(df) > 2:
                kf = KalmanFilter(
                    initial_state_mean=df[['Lat', 'Lon']].values[0],
                    observation_covariance=np.eye(2)*0.0001,
                    transition_covariance=np.eye(2)*1e-5
                )
                smoothed_means, _ = kf.smooth(df[['Lat', 'Lon']].values)
                lat_kf, lon_kf = smoothed_means[-1]
                if np.isnan(lat_kf): lat_kf = None
                if np.isnan(lon_kf): lon_kf = None
        except Exception as kf_err:
            print(f"Kalman Error: {kf_err}")
        
        results = {
            "median": {"lat": lat_med, "lon": lon_med},
            "kalman": {"lat": lat_kf, "lon": lon_kf},
            "lstm": None,
            "cnn": None
        }
        
        if len(df) > LOOK_BACK + 5:
            # Preprocess
            df['Lat_Smooth'] = df['Lat'].rolling(window=3, min_periods=1).mean()
            df['Lon_Smooth'] = df['Lon'].rolling(window=3, min_periods=1).mean()
            
            scaler = MinMaxScaler()
            scaled_vals = scaler.fit_transform(df[['Lat_Smooth', 'Lon_Smooth']].fillna(method='bfill').values)
            
            X_seq, y_target = [], []
            X_img = []
            
            for i in range(len(scaled_vals) - LOOK_BACK):
                seq = scaled_vals[i:i+LOOK_BACK]
                target = scaled_vals[i+LOOK_BACK]
                X_seq.append(seq)
                X_img.append(generate_occupancy_grid(seq, IMG_SIZE))
                y_target.append(target)
                
            if len(X_seq) > 0:
                X_lstm = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
                X_cnn = torch.tensor(np.array(X_img), dtype=torch.float32).to(device)
                y_t = torch.tensor(np.array(y_target), dtype=torch.float32).to(device)
                
                # --- LSTM ---
                try:
                    lstm_model = LSTMModel().to(device)
                    opt = optim.Adam(lstm_model.parameters(), lr=0.01)
                    crit = nn.MSELoss()
                    lstm_model.train()
                    for _ in range(10): # Fast training
                        opt.zero_grad()
                        out = lstm_model(X_lstm)
                        loss = crit(out, y_t)
                        loss.backward()
                        opt.step()
                        
                    lstm_model.eval()
                    last_seq = scaled_vals[-LOOK_BACK:]
                    last_in_lstm = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_lstm = lstm_model(last_in_lstm).detach().cpu().numpy()
                    pred_lstm_real = scaler.inverse_transform(pred_lstm)[0]
                    results['lstm'] = {"lat": float(pred_lstm_real[0]), "lon": float(pred_lstm_real[1])}
                except Exception as lstm_e:
                    print(f"LSTM Error: {lstm_e}")

                # --- CNN ---
                try:
                    cnn_model = GPS_CNN().to(device)
                    opt_c = optim.Adam(cnn_model.parameters(), lr=0.01)
                    cnn_model.train()
                    for _ in range(10):
                        opt_c.zero_grad()
                        out = cnn_model(X_cnn)
                        loss = crit(out, y_t)
                        loss.backward()
                        opt_c.step()
                        
                    cnn_model.eval()
                    last_img = generate_occupancy_grid(last_seq, IMG_SIZE)
                    last_in_cnn = torch.tensor(last_img[np.newaxis, ...], dtype=torch.float32).to(device)
                    pred_cnn = cnn_model(last_in_cnn).detach().cpu().numpy()
                    pred_cnn_real = scaler.inverse_transform(pred_cnn)[0]
                    results['cnn'] = {"lat": float(pred_cnn_real[0]), "lon": float(pred_cnn_real[1])}
                except Exception as cnn_e:
                     print(f"CNN Error: {cnn_e}")

        # Ensure no NaNs in final result
        def sanitize(v):
            if v is None: return None
            if np.isnan(v['lat']) or np.isnan(v['lon']): return None
            return v
            
        results['median'] = sanitize(results['median'])
        results['kalman'] = sanitize(results['kalman'])
        
        return json.dumps(results, cls=SafeJSONEncoder)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        init_db()
    except:
        print("Warning: Database init failed (could be connection issue).")
    app.run(debug=True, port=5000)