import mysql.connector
import os

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            user=os.environ.get("DB_USER", "root"),
            password=os.environ.get("DB_PASSWORD", ""),
            database=os.environ.get("DB_NAME", "locnet_db")
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        # Try connecting without database to create it
        try:
             conn = mysql.connector.connect(
                host=os.environ.get("DB_HOST", "localhost"),
                user=os.environ.get("DB_USER", "root"),
                password=os.environ.get("DB_PASSWORD", "")
            )
             cursor = conn.cursor()
             cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.environ.get('DB_NAME', 'locnet_db')}")
             conn.database = os.environ.get("DB_NAME", "locnet_db")
        except mysql.connector.Error as err:
            print(f"Critical DB Init Error: {err}")
            return

    cursor = conn.cursor()
    
    # Dataset Metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT FALSE
    )
    """)
    
    # GPS Data Points
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gps_points (
        id INT AUTO_INCREMENT PRIMARY KEY,
        upload_id INT,
        lat DOUBLE,
        lon DOUBLE,
        hdop DOUBLE,
        timestamp TIMESTAMP,
        FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
