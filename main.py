import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.4

# Road related classes
road_classes = [
    'person','bicycle','car','motorcycle','bus','truck',
    'traffic light','stop sign','dog','cat','horse','sheep','cow'
]

# Load video
cap = cv2.VideoCapture("dash1.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("[INFO] Starting detection... Press q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    h, w, _ = frame.shape

    # Define front collision zone
    zone_x1 = int(w * 0.30)
    zone_x2 = int(w * 0.70)
    zone_y1 = int(h * 0.35)
    zone_y2 = h

    # Draw collision zone
    cv2.rectangle(frame,(zone_x1,zone_y1),(zone_x2,zone_y2),(0,0,255),2)
    cv2.putText(frame,"Collision Zone",(zone_x1,zone_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # Run YOLO detection
    results = model(frame)

    df = results.pandas().xyxy[0]
    filtered_df = df[df['name'].isin(road_classes)]

    fcw_warning = False
    collision_warning = False
    emergency_brake = False

    for _, row in filtered_df.iterrows():

        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        label = row['name']

        width = x2 - x1
        height = y2 - y1
        area = width * height

        # -------- Distance Simulation Added --------
        distance = round(8000 / area, 2)
        # -------------------------------------------

        # Draw bounding box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Show object name + simulated distance
        cv2.putText(frame,
                    f"{label.upper()} {distance}m",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        # Object center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        inside_zone = (zone_x1 < cx < zone_x2) and (zone_y1 < cy < zone_y2)

        if label in ["car","truck","bus","motorcycle","person"]:

            if inside_zone and area > 90000:
                emergency_brake = True

            elif inside_zone and area > 60000:
                collision_warning = True

            elif inside_zone and area > 35000:
                fcw_warning = True

    # Forward Collision Warning
    if fcw_warning:
        cv2.putText(frame,
                    "FORWARD COLLISION WARNING",
                    (50,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)

    # Possible Collision Warning
    if collision_warning:
        cv2.putText(frame,
                    "POSSIBLE COLLISION DETECTED",
                    (50,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    # Emergency Brake
    if emergency_brake:
        cv2.putText(frame,
                    "EMERGENCY BRAKE ACTIVATED!",
                    (50,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    4)

        # Simulate braking delay
        cv2.waitKey(200)

    cv2.imshow("Road & Pedestrian Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()