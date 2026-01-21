import cv2
from deepface import DeepFace

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting camera... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            # Analyze face
            result = DeepFace.analyze(
                face_img,
                actions=["age", "emotion"],
                enforce_detection=False
            )

            age = int(result[0]["age"])
            emotion = result[0]["dominant_emotion"]

            label = f"Age: {age}, Emotion: {emotion}"

        except:
            label = "Detecting..."

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw label background
        cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 255, 0), -1)

        # Put text
        cv2.putText(frame, label, (x+5, y-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Age & Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
