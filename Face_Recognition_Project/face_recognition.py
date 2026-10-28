import cv2
import csv
import os
from datetime import datetime


# ==========================================
# 1. Face Detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# 2. Load Face Model
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("face_model.yml")


# ==========================================
# 3. Load Employee Information
# ==========================================

employees = {}

with open(
    "employees.txt",
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        label, employee_id, name = line.strip().split(",", 2)

        employees[int(label)] = {
            "id": employee_id,
            "name": name
        }


print()
print("Registered Employees:")

for label, employee in employees.items():

    print(
        employee["id"],
        "-",
        employee["name"]
    )


# ==========================================
# 4. Attendance File
# ==========================================

attendance_file = "attendance.csv"


# Create attendance.csv if not exists

if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Employee ID",
            "Employee Name",
            "Date",
            "Time"
        ])


# ==========================================
# 5. Already Marked Today
# ==========================================

today = datetime.now().strftime("%Y-%m-%d")

marked_today = set()


with open(
    attendance_file,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["Date"] == today:

            marked_today.add(
                row["Employee ID"]
            )


# ==========================================
# 6. Attendance Function
# ==========================================

def mark_attendance(employee_id, employee_name):

    # Already marked today
    if employee_id in marked_today:

        return False


    now = datetime.now()

    date = now.strftime("%Y-%m-%d")

    time = now.strftime("%H:%M:%S")


    # Save attendance

    with open(
        attendance_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            employee_id,
            employee_name,
            date,
            time
        ])


    # Add to today's marked list

    marked_today.add(employee_id)


    print()
    print("======================================")
    print("ATTENDANCE MARKED")
    print("Employee ID :", employee_id)
    print("Employee    :", employee_name)
    print("Date        :", date)
    print("Time        :", time)
    print("======================================")


    return True


# ==========================================
# 7. Start Webcam
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Camera isn't opening.")

    exit()


print()
print("======================================")
print("Employee Face Recognition Started")
print("======================================")
print("Press Q to Exit")
print()


# ==========================================
# 8. Live Recognition
# ==========================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Frame is not being read..")

        break


    # Convert to grayscale

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    # ======================================
    # Process every detected face
    # ======================================

    for (x, y, w, h) in detected_faces:

        face = gray[
            y:y + h,
            x:x + w
        ]


        # Same size as training

        face = cv2.resize(
            face,
            (200, 200)
        )


        # ==================================
        # Predict
        # ==================================

        label, confidence = recognizer.predict(
            face
        )


        # ==================================
        # Recognized?
        # ==================================

        if confidence < 70 and label in employees:

            employee_id = employees[label]["id"]

            employee_name = employees[label]["name"]

            # Mark attendance

            mark_attendance(
                employee_id,
                employee_name
            )

        else:

            employee_id = "Unknown"

            employee_name = "Unknown"


        # ==================================
        # Draw Rectangle
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ==================================
        # Name
        # ==================================

        cv2.putText(
            frame,
            employee_name,
            (x, y - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==================================
        # Employee ID
        # ==================================

        cv2.putText(
            frame,
            f"ID: {employee_id}",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==================================
        # Score
        # ==================================

        cv2.putText(
            frame,
            f"Score: {confidence:.1f}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ======================================
    # Show Camera
    # ======================================

    cv2.imshow(
        "Employee Face Recognition",
        frame
    )


    # ======================================
    # Q = Exit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 9. Close
# ==========================================

cap.release()

cv2.destroyAllWindows()

print()
print("Recognition stopped.")