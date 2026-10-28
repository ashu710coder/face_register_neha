import cv2
import os
import numpy as np


# ==========================================
# 1. Face Detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# 2. Training Data
# ==========================================

faces = []
labels = []

label_names = {}
employee_ids = {}

current_label = 0

known_faces_folder = "known_faces"


# ==========================================
# 3. Check Folder
# ==========================================

if not os.path.exists(known_faces_folder):

    print("ERROR: known_faces folder nahi mila.")
    exit()


# ==========================================
# 4. Read Employee Folders
# ==========================================

for folder_name in os.listdir(known_faces_folder):

    person_folder = os.path.join(
        known_faces_folder,
        folder_name
    )

    if not os.path.isdir(person_folder):
        continue


    # ======================================
    # Folder format:
    # 202060_Ashutosh Pal
    # ======================================

    if "_" not in folder_name:

        print(
            f"Invalid folder name: {folder_name}"
        )

        continue


    employee_id, employee_name = folder_name.split(
        "_",
        1
    )


    print()
    print(
        f"Training: {employee_id} - {employee_name}"
    )


    # Save label information

    label_names[current_label] = employee_name

    employee_ids[current_label] = employee_id


    # ======================================
    # Read Images
    # ======================================

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )


        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            continue


        # Read image

        image = cv2.imread(
            image_path
        )


        if image is None:

            print(
                f"Image read nahi hui: {image_name}"
            )

            continue


        # ==================================
        # Grayscale
        # ==================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ==================================
        # Detect Face
        # ==================================

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        if len(detected_faces) == 0:

            print(
                f"Face nahi mila: {image_name}"
            )

            continue


        # ==================================
        # Extract Face
        # ==================================

        for (x, y, w, h) in detected_faces:

            face = gray[
                y:y + h,
                x:x + w
            ]


            face = cv2.resize(
                face,
                (200, 200)
            )


            faces.append(face)

            labels.append(
                current_label
            )


            print(
                f"  Added: {image_name}"
            )


    current_label += 1


# ==========================================
# 5. Check Data
# ==========================================

if len(faces) == 0:

    print(
        "\nERROR: Koi face training ke liye nahi mila."
    )

    exit()


# ==========================================
# 6. Convert Labels
# ==========================================

labels = np.array(labels)


# ==========================================
# 7. Create Recognizer
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()


# ==========================================
# 8. Train
# ==========================================

print()
print("Training model...")

recognizer.train(
    faces,
    labels
)


# ==========================================
# 9. Save Model
# ==========================================

recognizer.write(
    "face_model.yml"
)


# ==========================================
# 10. Save Employee Information
# ==========================================

with open(
    "employees.txt",
    "w",
    encoding="utf-8"
) as file:

    for label in label_names:

        name = label_names[label]

        employee_id = employee_ids[label]

        file.write(
            f"{label},{employee_id},{name}\n"
        )


# ==========================================
# 11. Final Output
# ==========================================

print()
print("======================================")
print("Training completed successfully!")
print("======================================")

print(
    "Total training faces:",
    len(faces)
)

print(
    "Total employees:",
    len(label_names)
)

print()
print("Created:")
print("1. face_model.yml")
print("2. employees.txt")