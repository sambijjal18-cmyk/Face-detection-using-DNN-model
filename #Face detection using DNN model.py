#Face detection using DNN model
import cv2

# Load DNN model

model = cv2.dnn.readNetFromCaffe(
    r"C:\Users\sambi\OneDrive\Documents\Face detection_models\deploy.prototxt",
    r"C:\Users\sambi\OneDrive\Documents\Face detection_models\res10_300x300_ssd_iter_140000.caffemodel"
)

# Open webcam

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions

    h, w = frame.shape[:2]

    # Convert frame into blob

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Pass blob into network

    model.setInput(blob)

    # Perform detection

    detections = model.forward()

    # Loop through detections

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Filter weak detections

        if confidence > 0.5:

            # Get coordinates

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            (x1, y1, x2, y2) = box.astype("int")

            # Draw bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display confidence

            text = f"{confidence * 100:.2f}%"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Display output

    cv2.imshow("DNN Face Detection", frame)

    # Press q to exit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources

cap.release()
cv2.destroyAllWindows()