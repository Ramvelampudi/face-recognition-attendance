import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and do face encoding
Ram_image = face_recognition.load_image_file(r"Faces\Ram.png")
Ram_encoding = face_recognition.face_encodings(Ram_image)[0]

Ashish_image = face_recognition.load_image_file(r"Faces\Ashish.jpeg")
Ashish_encoding = face_recognition.face_encodings(Ashish_image)[0]

Jaya_image = face_recognition.load_image_file(r"Faces\Jaya.jpeg")
Jaya_encoding = face_recognition.face_encodings(Jaya_image)[0]

Surya_image = face_recognition.load_image_file(r"Faces\Surya.jpeg")
Surya_encoding = face_recognition.face_encodings(Surya_image)[0]

Satyanarayana_image = face_recognition.load_image_file(r"Faces\Satyanarayana.jpeg")
Satyanarayana_encoding = face_recognition.face_encodings(Satyanarayana_image)[0]

# Saving names and encodings
known_face_encodings = [Ram_encoding, Ashish_encoding, Jaya_encoding, Surya_encoding, Satyanarayana_encoding]
known_face_names = ["Ram", "Ashish", "Jaya", "Surya", "Satyanarayana"]

# List of expected students
students = known_face_names.copy()

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

# Write on csv (append mode)
with open(f"{current_date}.csv", "a", newline="") as f:
    lnwriter = csv.writer(f)

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Initialize name list for this frame
        names_in_frame = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            name = "Unknown"  # Default name when the face is not recognized

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            names_in_frame.append(name)

            # Add the text if a person is present
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerofText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            current_time = now.strftime("%I:%M %p")  # Get the current time in 12-hour format with AM/PM
            text = name + " Present " 
            cv2.putText(frame, text, bottomLeftCornerofText, font, fontScale, fontColor, thickness, lineType)

        # Handle unknown faces by removing the last recognized name
        if "Unknown" in names_in_frame and len(names_in_frame) == 1:
            students.clear()

        for name in names_in_frame:
            if name in students:
                students.remove(name)
                lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
