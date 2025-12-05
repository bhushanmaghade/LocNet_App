# LocNet: GNSS Position Optimizer

**LocNet** is a backend API service designed to estimate the **True Position** from noisy GNSS (Global Navigation Satellite System) data. It exposes REST endpoints to process GPS data using statistical filters and Deep Learning models.

---

## Overview

GPS sensors often suffer from noise, drift, and signal multipath errors. **LocNet** addresses these issues by employing four distinct approaches to refine the coordinate data:

1.  **Statistical Median**: A robust baseline that filters out random noise.
2.  **Kalman Filter**: A recursive mathematical algorithm for state estimation.
3.  **LSTM (Long Short-Term Memory)**: A Recurrent Neural Network (RNN) for sequential pattern learning.
4.  **CNN (Convolutional Neural Network)**: A computer vision approach using Time-Encoded Occupancy Grids.

---

## Technical Stack

-   **Framework**: Flask (Python)
-   **Database**: MySQL
-   **Machine Learning**: PyTorch, Scikit-Learn
-   **Math/Processing**: NumPy, Pandas, PyKalman

---

## Installation

### Prerequisites
-   **Python 3.8+**
-   **MySQL Server** running locally or remotely.

### Steps

1.  **Clone the Repository**
    ```bash
    cd path/to/GPS_Predictor
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Database**
    Set the following environment variables (defaults provided below):
    -   `DB_HOST` (default: localhost)
    -   `DB_USER` (default: root)
    -   `DB_PASSWORD` (default: empty)
    -   `DB_NAME` (default: locnet_db)

---

## Usage

1.  **Start the Server**
    ```bash
    python app.py
    ```
    The server will start on `http://localhost:5000`.

2.  **API Endpoints**

    ### `POST /upload`
    Upload a CSV file containing GPS data.
    -   **Body**: `form-data` with key `file` (CSV file).
    -   **Response**: JSON with `upload_id` and status.

    ### `GET /predict/<upload_id>`
    Trigger processing and prediction for a specific dataset.
    -   **Response**: JSON containing optimized coordinates from all models.
        ```json
        {
          "median": {"lat": ..., "lon": ...},
          "kalman": {"lat": ..., "lon": ...},
          "lstm": {"lat": ..., "lon": ...},
          "cnn": {"lat": ..., "lon": ...}
        }
        ```

---

## Methodologies

### 1. Statistical Median
Calculates the geometric median to ignore outliers.

### 2. Kalman Filter
Standard Kalman Filter to smooth the trajectory.

### 3. LSTM (Deep Learning)
Treats the GPS path as a time-series sequence. Trains on the uploaded trajectory to predict the final "true" position.

### 4. CNN (Computer Vision)
Converts the trajectory into a **Time-Encoded Occupancy Grid** image and uses a CNN to predict the coordinate offset.
