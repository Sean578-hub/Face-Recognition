import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = DeepFace.analyze(frame, actions = ["age", "gender", "emotion"], enforce_detection = False)
    age = result[0]["age"]
    gender = result[0]["dominant_gender"]
    emotion = result[0]["dominant_emotion"]

    cv2.putText(frame, f"Age: {age}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gender: {gender}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("DeepFace Real-Time", frame)

    key = cv2.waitKey(1) & 0xFF
    if key is ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
