import cv2
import csv
import os
import tkinter as tk
from tkinter import messagebox
from datetime import datetime


# ==========================================
# SETTINGS
# ==========================================

MODEL_FILE = "face_model.yml"
EMPLOYEE_FILE = "employees.txt"
ATTENDANCE_FILE = "attendance.csv"

CONFIDENCE_THRESHOLD = 55


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(MODEL_FILE)


# ==========================================
# LOAD EMPLOYEES
# ==========================================

employees = {}

with open(
    EMPLOYEE_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        label, employee_id, name = line.strip().split(",", 2)

        employees[int(label)] = {
            "id": employee_id,
            "name": name
        }


# ==========================================
# CREATE ATTENDANCE FILE
# ==========================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Employee ID",
            "Employee Name",
            "Date",
            "IN Time",
            "OUT Time",
            "Total Hours"
        ])


# ==========================================
# GET TODAY'S RECORD
# ==========================================

def get_today_record(employee_id):

    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(ATTENDANCE_FILE):
        return None

    with open(
        ATTENDANCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["Employee ID"] == employee_id
                and row["Date"] == today
            ):

                return row

    return None


# ==========================================
# UPDATE CSV RECORD
# ==========================================

def update_attendance(
    employee_id,
    employee_name,
    mode
):

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    rows = []

    record_found = False

    with open(
        ATTENDANCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["Employee ID"] == employee_id
                and row["Date"] == today
            ):

                record_found = True

                # ==========================
                # IN
                # ==========================

                if mode == "IN":

                    if row["IN Time"] != "":

                        return (
                            False,
                            "IN already marked today."
                        )

                    row["IN Time"] = current_time


                # ==========================
                # OUT
                # ==========================

                elif mode == "OUT":

                    if row["IN Time"] == "":

                        return (
                            False,
                            "First mark IN."
                        )

                    if row["OUT Time"] != "":

                        return (
                            False,
                            "OUT already marked today."
                        )

                    row["OUT Time"] = current_time


                    # Calculate working hours

                    in_time = datetime.strptime(
                        row["IN Time"],
                        "%H:%M:%S"
                    )

                    out_time = datetime.strptime(
                        current_time,
                        "%H:%M:%S"
                    )

                    difference = (
                        out_time - in_time
                    )

                    total_seconds = int(
                        difference.total_seconds()
                    )

                    hours = total_seconds // 3600

                    minutes = (
                        total_seconds % 3600
                    ) // 60

                    seconds = (
                        total_seconds % 60
                    )

                    row["Total Hours"] = (
                        f"{hours:02d}:"
                        f"{minutes:02d}:"
                        f"{seconds:02d}"
                    )


            rows.append(row)


    # ======================================
    # Create New IN Record
    # ======================================

    if not record_found:

        if mode == "OUT":

            return (
                False,
                "First mark IN."
            )

        rows.append({

            "Employee ID": employee_id,
            "Employee Name": employee_name,
            "Date": today,
            "IN Time": current_time,
            "OUT Time": "",
            "Total Hours": ""
        })


    # ======================================
    # Save CSV
    # ======================================

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Employee ID",
            "Employee Name",
            "Date",
            "IN Time",
            "OUT Time",
            "Total Hours"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


    return (
        True,
        f"{mode} marked successfully at {current_time}"
    )


# ==========================================
# FACE RECOGNITION
# ==========================================

def start_camera(mode):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Camera open nahi ho raha."
        )

        return


    recognized_employee = None

    print(f"{mode} mode started.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]


            face = cv2.resize(
                face,
                (200, 200)
            )


            label, confidence = recognizer.predict(
                face
            )


            # ==============================
            # Recognized
            # ==============================

            if (
                confidence < CONFIDENCE_THRESHOLD
                and label in employees
            ):

                employee = employees[label]

                employee_id = employee["id"]

                employee_name = employee["name"]

                display_text = (
                    f"{employee_name} | "
                    f"ID: {employee_id}"
                )


                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    display_text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Score: {confidence:.1f}",
                    (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                # ==========================
                # Mark Attendance
                # ==========================

                if recognized_employee != employee_id:

                    success, message = update_attendance(
                        employee_id,
                        employee_name,
                        mode
                    )

                    print(message)


                    if success:

                        messagebox.showinfo(
                            "Attendance",
                            f"{mode} successful!\n\n"
                            f"Name: {employee_name}\n"
                            f"Employee ID: {employee_id}"
                        )

                    else:

                        messagebox.showwarning(
                            "Attendance",
                            message
                        )


                    recognized_employee = employee_id


            else:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


        cv2.imshow(
            f"Face Attendance - {mode}",
            frame
        )


        # Press Q to close camera

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    cap.release()

    cv2.destroyAllWindows()


# ==========================================
# BUTTON FUNCTIONS
# ==========================================

def in_button():

    start_camera("IN")


def out_button():

    start_camera("OUT")


# ==========================================
# GUI
# ==========================================

root = tk.Tk()

root.title(
    "Face Attendance System"
)

root.geometry(
    "600x400"
)


# Title

title = tk.Label(
    root,
    text="FACE ATTENDANCE SYSTEM",
    font=("Arial", 22, "bold")
)

title.pack(
    pady=30
)


# Instruction

instruction = tk.Label(
    root,
    text="Select IN or OUT",
    font=("Arial", 14)
)

instruction.pack(
    pady=10
)


# ==========================================
# IN BUTTON
# ==========================================

in_btn = tk.Button(
    root,
    text="IN",
    font=("Arial", 18, "bold"),
    width=12,
    height=2,
    command=in_button
)

in_btn.pack(
    side="left",
    padx=60,
    pady=40
)


# ==========================================
# OUT BUTTON
# ==========================================

out_btn = tk.Button(
    root,
    text="OUT",
    font=("Arial", 18, "bold"),
    width=12,
    height=2,
    command=out_button
)

out_btn.pack(
    side="right",
    padx=60,
    pady=40
)


# ==========================================
# Footer
# ==========================================

footer = tk.Label(
    root,
    text="Press Q to close camera",
    font=("Arial", 10)
)

footer.pack(
    side="bottom",
    pady=20
)


# ==========================================
# Start GUI
# ==========================================

root.mainloop()