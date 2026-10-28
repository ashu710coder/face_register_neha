import cv2

# Face detector load
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Webcam start
cap = cv2.VideoCapture(0)

while True:

    # Webcam se frame read
    ret, frame = cap.read()

    if not ret:
        print("Camera is not opening.")
        break

    # Frame ko grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Faces detect karo
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Har face ke around rectangle
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Webcam window show karo
    cv2.imshow("Face Detection", frame)

    # Q press karne par program band
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Camera release
cap.release()
cv2.destroyAllWindows()