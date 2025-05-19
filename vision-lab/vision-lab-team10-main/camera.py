import cv2

cam = cv2.VideoCapture("/dev/video4")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cam.set(cv2.CAP_PROP_FPS, 60)

if not cam.isOpened():
    print("Error: Could not open video device.")
    exit()

print("Starting camera stream...")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
        
        cv2.imshow('Realsense Camera', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()
    print("Camera stream stopped.")
