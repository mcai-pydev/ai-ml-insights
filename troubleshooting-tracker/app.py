from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Initialize SQLite database
DB_FILE = "troubleshooting.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                problem TEXT,
                symptoms TEXT,
                what_didnt_work TEXT,
                what_worked TEXT,
                resources TEXT
            )
        ''')
        conn.commit()

init_db()

# Route for the frontend
@app.route("/")
def index():
    return render_template("index.html")

# API to save troubleshooting logs
@app.route("/save_log", methods=["POST"])
def save_log():
    data = request.json
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    problem = data.get("problem")
    symptoms = data.get("symptoms")
    what_didnt_work = data.get("what_didnt_work")
    what_worked = data.get("what_worked")
    resources = data.get("resources")

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            INSERT INTO logs (timestamp, problem, symptoms, what_didnt_work, what_worked, resources)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, problem, symptoms, what_didnt_work, what_worked, resources))
        conn.commit()

    return jsonify({"message": "Log saved successfully!"}), 200

# API to fetch troubleshooting logs
@app.route("/logs", methods=["GET"])
def fetch_logs():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute('SELECT * FROM logs ORDER BY timestamp DESC')
        logs = [{"id": row[0], "timestamp": row[1], "problem": row[2], "symptoms": row[3],
                 "what_didnt_work": row[4], "what_worked": row[5], "resources": row[6]} for row in cursor.fetchall()]

    return jsonify(logs), 200

if __name__ == "__main__":
    app.run(debug=True)

