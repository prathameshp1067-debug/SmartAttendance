import cv2
import sqlite3
import numpy as np
import face_recognition
from datetime import datetime
from openpyxl import Workbook, load_workbook
from flask import Flask, render_template, Response
import os
import mimetypes

mimetypes.init(files=[])

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

# ============================
# Auto Camera Selection
# ============================
def get_camera_source():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Using laptop webcam")
        return cap
    cap.release()

    ip_cap = cv2.VideoCapture("rtsp://192.168.1.50:554/stream")  # adjust URL
    if ip_cap.isOpened():
        print("Using IP CCTV camera")
        return ip_cap
    ip_cap.release()

    raise Exception("No camera source available")

camera_cap = get_camera_source()

# ============================
# Load Known Faces
# ============================
DATASET_PATH = r"D:/attend/dataset"
known_face_encodings, known_face_names, known_face_rollnos = [], [], []

for student_name in os.listdir(DATASET_PATH):
    student_folder = os.path.join(DATASET_PATH, student_name)
    if os.path.isdir(student_folder):
        for rollno in os.listdir(student_folder):
            roll_folder = os.path.join(student_folder, rollno)
            if os.path.isdir(roll_folder):
                for img in os.listdir(roll_folder):
                    path = os.path.join(roll_folder, img)
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        image = face_recognition.load_image_file(path)
                        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                        encodings = face_recognition.face_encodings(small_image, num_jitters=1)
                        if len(encodings) == 0:
                            print(f"No face found in {path}, skipping...")
                            continue
                        encoding = encodings[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(student_name)
                        known_face_rollnos.append(rollno)

# ============================
# Master Roll List (1–70)
# ============================
all_students = [str(i) for i in range(1, 71)]

# ============================
# Database Setup
# ============================
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS attendance")
cursor.execute("""
CREATE TABLE attendance (
    name TEXT,
    rollno TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

attendance_marked = set()

def mark_attendance(name, rollno):
    global attendance_marked
    key = f"{name}_{rollno}"
    if key in attendance_marked:
        return

    now = datetime.now()
    date_string = now.strftime("%d-%m-%y")
    time_string = now.strftime("%I:%M:%S %p")

    cursor.execute("INSERT INTO attendance (name, rollno, date, time) VALUES (?, ?, ?, ?)",
                   (name, rollno, date_string, time_string))
    conn.commit()

    filename = os.path.join(os.path.dirname(__file__), "attendance.xlsx")
    if not os.path.exists(filename):
        wb = Workbook()
        wb.remove(wb.active)
        ws_present = wb.create_sheet("Present")
        ws_absent = wb.create_sheet("Absent")
        ws_present.append(["Name", "Roll No", "Date", "Time", "Status"])
        ws_absent.append(["Roll No"])
        wb.save(filename)

    wb = load_workbook(filename)
    ws_present = wb["Present"]
    ws_present.append([name, rollno, date_string, time_string, "Present"])
    wb.save(filename)

    print(f"Attendance marked for: {name} ({rollno})")
    attendance_marked.add(key)

def save_absentees():
    filename = os.path.join(os.path.dirname(__file__), "attendance.xlsx")
    wb = load_workbook(filename)
    if "Absent" not in wb.sheetnames:
        ws_absent = wb.create_sheet("Absent")
        ws_absent.append(["Roll No"])
    else:
        ws_absent = wb["Absent"]
        ws_absent.delete_rows(2, ws_absent.max_row)

    cursor.execute("SELECT rollno FROM attendance")
    present_rolls = [row[0] for row in cursor.fetchall()]
    absent_rolls = [roll for roll in all_students if roll not in present_rolls]

    for roll in absent_rolls:
        ws_absent.append([roll])
    wb.save(filename)
    print("Absent roll numbers saved.")

# ============================
# Camera Generator
# ============================
def gen_frames():
    process_frame = True
    while True:
        success, frame = camera_cap.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_frame:
            locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            encodings = face_recognition.face_encodings(rgb_small_frame, locations)

            for encoding, (top, right, bottom, left) in zip(encodings, locations):
                matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, encoding)

                name, rollno = "Unknown", "N/A"
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        rollno = known_face_rollnos[best_match_index]
                        mark_attendance(name, rollno)

                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({rollno})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        process_frame = not process_frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ============================
# Routes
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin')
def admin_page():
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    save_absentees()
    return render_template('admin.html', records=records)

@app.route('/absent')
def absent_page():
    cursor.execute("SELECT rollno FROM attendance")
    present_rolls = [row[0] for row in cursor.fetchall()]
    absent_rolls = [roll for roll in all_students if roll not in present_rolls]
    return render_template('absent.html', absent_rolls=absent_rolls)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)