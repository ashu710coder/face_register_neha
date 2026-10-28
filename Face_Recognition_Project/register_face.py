import cv2
import os

# ==========================================
# 1. Employee Details
# ==========================================

employee_id = input("Enter Employee ID: ").strip()
employee_name = input("Enter Employee Name: ").strip()


# ==========================================
# 2. Validate Input
# ==========================================

if employee_id == "" or employee_name == "":
    print("Both the employee ID and name are required.")
    exit()


# ==========================================
# 3. Create Employee Folder
# ==========================================

folder_name = f"{employee_id}_{employee_name}"

save_folder = os.path.join(
    "known_faces",
    folder_name
)

os.makedirs(
    save_folder,
    exist_ok=True
)

print()
print("Employee:", employee_name)
print("Employee ID:", employee_id)
print("Images will save in:", save_folder)


# ==========================================
# 4. Face Detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# 5. Start Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera isn't opening..")
    exit()


print()
print("========================================")
print("Face Registration Started")
print("Look at the camera")
print("Please move your face slightly")
print("Press Q to cancel")
print("========================================")


# ==========================================
# 6. Capture Settings
# ==========================================

total_images = 20
image_count = 0


# ==========================================
# 7. Capture Face Images
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame is not being read.")
        break


    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # ======================================
    # Process Face
    # ======================================

    if len(faces) > 0:

        # First detected face
        x, y, w, h = faces[0]


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Capture image every few frames
        if image_count < total_images:

            # Crop face
            face = gray[
                y:y + h,
                x:x + w
            ]


            # Resize face
            face = cv2.resize(
                face,
                (200, 200)
            )


            # File name
            image_path = os.path.join(
                save_folder,
                f"{image_count + 1}.jpg"
            )


            # Save image
            cv2.imwrite(
                image_path,
                face
            )


            image_count += 1

            print(
                f"Face captured: "
                f"{image_count}/{total_images}"
            )


    # ======================================
    # Display Progress
    # ======================================

    progress_text = (
        f"Registered: "
        f"{image_count}/{total_images}"
    )


    cv2.putText(
        frame,
        progress_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        employee_name,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # Show camera
    cv2.imshow(
        "Face Registration",
        frame
    )


    # ======================================
    # Registration Complete
    # ======================================

    if image_count >= total_images:

        print()
        print("========================================")
        print("Face Registration Successful!")
        print("========================================")
        print("Employee Name:", employee_name)
        print("Employee ID:", employee_id)
        print("Images Captured:", image_count)
        print("Folder:", save_folder)
        print("========================================")

        break


    # Q = Exit
    if cv2.waitKey(100) & 0xFF == ord("q"):

        print("Registration cancelled.")
        break


# ==========================================
# 8. Close Camera
# ==========================================

cap.release()

cv2.destroyAllWindows()
