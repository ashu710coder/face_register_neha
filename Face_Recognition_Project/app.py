from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import os
import csv
from datetime import datetime


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# FILES / FOLDERS
# ============================================================

MODEL_FILE = "face_model.yml"

EMPLOYEE_FILE = "employees.txt"

ATTENDANCE_FILE = "attendance.csv"

KNOWN_FACES_FOLDER = "known_faces"


TOTAL_FACE_SAMPLES = 25


# Recognition threshold
# Lower value = stricter
CONFIDENCE_THRESHOLD = 55


# ============================================================
# CREATE FOLDER
# ============================================================

os.makedirs(
    KNOWN_FACES_FOLDER,
    exist_ok=True
)


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Check detector

if face_cascade.empty():

    print(
        "ERROR: Haar Cascade load nahi hua."
    )

else:

    print(
        "Face detector loaded successfully."
    )


# ============================================================
# FACE RECOGNIZER
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_loaded = False


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

if os.path.exists(MODEL_FILE):

    try:

        recognizer.read(
            MODEL_FILE
        )

        model_loaded = True

        print(
            "Face model loaded successfully."
        )

    except Exception as e:

        print(
            "Model loading error:",
            e
        )


# ============================================================
# LOAD EMPLOYEES
# ============================================================

def load_employees():

    employees = {}


    if not os.path.exists(
        EMPLOYEE_FILE
    ):

        return employees


    with open(
        EMPLOYEE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()


            if not line:

                continue


            parts = line.split(
                ",",
                2
            )


            if len(parts) != 3:

                continue


            try:

                label = int(
                    parts[0]
                )

            except:

                continue


            employees[label] = {

                "id": parts[1],

                "name": parts[2]

            }


    return employees


# Global employee data

employees = load_employees()


# ============================================================
# ATTENDANCE FILE
# ============================================================

def create_attendance_file():

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow([

                "Employee ID",

                "Employee Name",

                "Date",

                "IN Time",

                "OUT Time",

                "Total Hours"

            ])


create_attendance_file()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SAVE EMPLOYEE
# ============================================================

def save_employee(
    employee_id,
    employee_name
):

    global employees


    employees = load_employees()


    # Check existing employee

    for label, employee in employees.items():

        if employee["id"] == employee_id:

            return label


    # New label

    if employees:

        new_label = max(
            employees.keys()
        ) + 1

    else:

        new_label = 0


    with open(
        EMPLOYEE_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(

            f"{new_label},"
            f"{employee_id},"
            f"{employee_name}\n"

        )


    employees = load_employees()


    return new_label


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global recognizer

    global model_loaded

    global employees


    print()
    print("====================================")
    print("FACE TRAINING STARTED")
    print("====================================")


    faces = []

    labels = []


    employees = load_employees()


    if not employees:

        print(
            "No employees available."
        )

        return False


    # ========================================================
    # EACH EMPLOYEE
    # ========================================================

    for label, employee in employees.items():

        employee_id = employee["id"]

        employee_name = employee["name"]


        safe_name = (

            employee_name
            .replace("/", "_")
            .replace("\\", "_")

        )


        folder_name = (

            f"{employee_id}_"
            f"{safe_name}"

        )


        folder_path = os.path.join(

            KNOWN_FACES_FOLDER,

            folder_name

        )


        if not os.path.exists(
            folder_path
        ):

            continue


        image_files = [

            file

            for file
            in os.listdir(folder_path)

            if file.lower().endswith(

                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )

            )

        ]


        print(
            f"{employee_id} - "
            f"{employee_name}: "
            f"{len(image_files)} images"
        )


        # ====================================================
        # READ IMAGES
        # ====================================================

        for image_file in image_files:

            image_path = os.path.join(

                folder_path,

                image_file

            )


            image = cv2.imread(

                image_path,

                cv2.IMREAD_GRAYSCALE

            )


            if image is None:

                continue


            image = cv2.resize(

                image,

                (200, 200)

            )


            faces.append(
                image
            )

            labels.append(
                label
            )


    # ========================================================
    # NO DATA
    # ========================================================

    if not faces:

        print(
            "No training images found."
        )

        return False


    # ========================================================
    # TRAIN
    # ========================================================

    new_recognizer = (

        cv2.face.LBPHFaceRecognizer_create()

    )


    new_recognizer.train(

        faces,

        np.array(labels)

    )


    # ========================================================
    # SAVE
    # ========================================================

    new_recognizer.write(

        MODEL_FILE

    )


    recognizer = new_recognizer

    model_loaded = True


    print(
        f"Total images: {len(faces)}"
    )

    print(
        "FACE TRAINING COMPLETED"
    )

    print(
        "===================================="
    )

    print()


    return True


# ============================================================
# REGISTER FACE
# ============================================================

@app.route(
    "/register",
    methods=["POST"]
)
def register_face():

    try:

        employee_id = (

            request.form.get(
                "employee_id",
                ""
            )
            .strip()

        )


        employee_name = (

            request.form.get(
                "employee_name",
                ""
            )
            .strip()

        )


        image_file = request.files.get(
            "image"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if not employee_id:

            return jsonify({

                "success": False,

                "message":
                    "Employee ID required."

            })


        if not employee_name:

            return jsonify({

                "success": False,

                "message":
                    "Employee Name required."

            })


        if image_file is None:

            return jsonify({

                "success": False,

                "message":
                    "Camera image nahi mili."

            })


        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = image_file.read()


        image_array = np.frombuffer(

            image_bytes,

            np.uint8

        )


        frame = cv2.imdecode(

            image_array,

            cv2.IMREAD_COLOR

        )


        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Image read nahi hui."

            })


        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )


        # ====================================================
        # HISTOGRAM EQUALIZATION
        # ====================================================

        gray = cv2.equalizeHist(
            gray
        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        faces = face_cascade.detectMultiScale(

            gray,

            scaleFactor=1.05,

            minNeighbors=4,

            minSize=(60, 60)

        )


        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            return jsonify({

                "success": False,

                "message":
                    "Face detect nahi hua. "
                    "Camera ke saamne clearly dekhiye."

            })


        # ====================================================
        # SELECT LARGEST FACE
        # ====================================================

        x, y, w, h = max(

            faces,

            key=lambda item:
                item[2] * item[3]

        )


        # ====================================================
        # PADDING
        # ====================================================

        padding = 20


        x1 = max(
            0,
            x - padding
        )


        y1 = max(
            0,
            y - padding
        )


        x2 = min(

            gray.shape[1],

            x + w + padding

        )


        y2 = min(

            gray.shape[0],

            y + h + padding

        )


        face_image = gray[

            y1:y2,

            x1:x2

        ]


        # ====================================================
        # RESIZE
        # ====================================================

        face_image = cv2.resize(

            face_image,

            (200, 200)

        )


        # ====================================================
        # EMPLOYEE FOLDER
        # ====================================================

        safe_name = (

            employee_name

            .replace("/", "_")

            .replace("\\", "_")

        )


        folder_name = (

            f"{employee_id}_"
            f"{safe_name}"

        )


        employee_folder = os.path.join(

            KNOWN_FACES_FOLDER,

            folder_name

        )


        os.makedirs(

            employee_folder,

            exist_ok=True

        )


        # ====================================================
        # SAVE EMPLOYEE
        # ====================================================

        save_employee(

            employee_id,

            employee_name

        )


        # ====================================================
        # GET CURRENT IMAGES
        # ====================================================

        image_files = [

            file

            for file
            in os.listdir(
                employee_folder
            )

            if file.lower().endswith(

                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )

            )

        ]


        image_number = (

            len(image_files) + 1

        )


        # ====================================================
        # SAVE FACE
        # ====================================================

        image_path = os.path.join(

            employee_folder,

            f"{image_number}.jpg"

        )


        cv2.imwrite(

            image_path,

            face_image

        )


        # ====================================================
        # TRAIN AFTER 25
        # ====================================================

        training_completed = False


        if (

            image_number >=
            TOTAL_FACE_SAMPLES

        ):

            training_completed = (
                train_model()
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "message":

                f"Face captured "
                f"{image_number}/"
                f"{TOTAL_FACE_SAMPLES}",

            "employee_id":
                employee_id,

            "employee_name":
                employee_name,

            "image_number":
                image_number,

            "training_completed":
                training_completed

        })


    except Exception as e:

        print(
            "REGISTER ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        })


# ============================================================
# RECOGNIZE FACE
# ============================================================

@app.route(
    "/recognize",
    methods=["POST"]
)
def recognize_face():

    try:

        global employees


        mode = request.form.get(
            "mode",
            ""
        )


        image_file = request.files.get(
            "image"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if image_file is None:

            return jsonify({

                "success": False,

                "message":
                    "Image nahi mili."

            })


        if not model_loaded:

            return jsonify({

                "success": False,

                "message":
                    "Face model available nahi hai. "
                    "Pehle registration complete karo."

            })


        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = image_file.read()


        image_array = np.frombuffer(

            image_bytes,

            np.uint8

        )


        frame = cv2.imdecode(

            image_array,

            cv2.IMREAD_COLOR

        )


        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Image read nahi hui."

            })


        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )


        gray = cv2.equalizeHist(
            gray
        )


        # ====================================================
        # DETECT FACE
        # ====================================================

        faces = face_cascade.detectMultiScale(

            gray,

            scaleFactor=1.05,

            minNeighbors=4,

            minSize=(60, 60)

        )


        if len(faces) == 0:

            return jsonify({

                "success": False,

                "message":
                    "Face detect nahi hua."

            })


        # ====================================================
        # LARGEST FACE
        # ====================================================

        x, y, w, h = max(

            faces,

            key=lambda item:
                item[2] * item[3]

        )


        face = gray[

            y:y+h,

            x:x+w

        ]


        face = cv2.resize(

            face,

            (200, 200)

        )


        # ====================================================
        # PREDICT
        # ====================================================

        label, confidence = (

            recognizer.predict(
                face
            )

        )


        print(

            f"Recognition: "
            f"Label={label}, "
            f"Score={confidence:.2f}"

        )


        employees = load_employees()


        # ====================================================
        # CHECK
        # ====================================================

        if (

            confidence >=
            CONFIDENCE_THRESHOLD

            or

            label not in employees

        ):

            return jsonify({

                "success": False,

                "message":
                    "Face recognized nahi hua.",

                "score":
                    round(
                        confidence,
                        2
                    )

            })


        # ====================================================
        # EMPLOYEE
        # ====================================================

        employee = employees[label]


        employee_id = employee["id"]

        employee_name = employee["name"]


        # ====================================================
        # MARK ATTENDANCE
        # ====================================================

        result = mark_attendance(

            employee_id,

            employee_name,

            mode

        )


        return jsonify({

            "success":
                result["success"],

            "message":
                result["message"],

            "employee_id":
                employee_id,

            "employee_name":
                employee_name,

            "score":
                round(
                    confidence,
                    2
                )

        })


    except Exception as e:

        print(
            "RECOGNITION ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        })


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(

    employee_id,

    employee_name,

    mode

):

    if mode not in [

        "IN",

        "OUT"

    ]:

        return {

            "success": False,

            "message":
                "Invalid mode."

        }


    today = datetime.now().strftime(

        "%Y-%m-%d"

    )


    current_time = datetime.now().strftime(

        "%H:%M:%S"

    )


    rows = []

    found = False


    # ========================================================
    # READ ATTENDANCE
    # ========================================================

    with open(

        ATTENDANCE_FILE,

        "r",

        encoding="utf-8"

    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            if (

                row["Employee ID"]
                == employee_id

                and

                row["Date"]
                == today

            ):

                found = True


                # ==========================================
                # IN
                # ==========================================

                if mode == "IN":

                    if row["IN Time"]:

                        return {

                            "success":
                                False,

                            "message":
                                "IN already marked today."

                        }


                    row["IN Time"] = (
                        current_time
                    )


                # ==========================================
                # OUT
                # ==========================================

                elif mode == "OUT":

                    if not row["IN Time"]:

                        return {

                            "success":
                                False,

                            "message":
                                "First mark IN."

                        }


                    if row["OUT Time"]:

                        return {

                            "success":
                                False,

                            "message":
                                "OUT already marked today."

                        }


                    row["OUT Time"] = (
                        current_time
                    )


                    # Calculate hours

                    in_time = datetime.strptime(

                        row["IN Time"],

                        "%H:%M:%S"

                    )


                    out_time = datetime.strptime(

                        current_time,

                        "%H:%M:%S"

                    )


                    difference = (

                        out_time -
                        in_time

                    )


                    total_seconds = int(

                        difference.total_seconds()

                    )


                    if total_seconds < 0:

                        total_seconds = 0


                    hours = (

                        total_seconds
                        // 3600

                    )


                    minutes = (

                        total_seconds
                        % 3600
                        // 60

                    )


                    seconds = (

                        total_seconds
                        % 60

                    )


                    row["Total Hours"] = (

                        f"{hours:02d}:"
                        f"{minutes:02d}:"
                        f"{seconds:02d}"

                    )


            rows.append(row)


    # ========================================================
    # CREATE NEW RECORD
    # ========================================================

    if not found:

        if mode == "OUT":

            return {

                "success":
                    False,

                "message":
                    "First mark IN."

            }


        rows.append({

            "Employee ID":
                employee_id,

            "Employee Name":
                employee_name,

            "Date":
                today,

            "IN Time":
                current_time,

            "OUT Time":
                "",

            "Total Hours":
                ""

        })


    # ========================================================
    # SAVE
    # ========================================================

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


    return {

        "success":
            True,

        "message":

            f"{mode} marked successfully "
            f"at {current_time}"

    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )