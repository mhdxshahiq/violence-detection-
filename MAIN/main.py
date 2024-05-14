import cv2
import torch
import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog

# Path to the alarm sound
path_alarm = r"C:\FILLE\COLLEGE\S6\#mini\target-detector-yolov5\Alarm\alarm.wav"
# Initializing pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Loading the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\FILLE\COLLEGE\S6\#mini\target-detector-yolov5\bestv5.pt')

target_classes = ['Violence', 'weapon', 'Non-violence', 'person']
count = 0
number_of_photos = 3
pts = []

# Global variable for video capture
cap = None

# Function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    if isinstance(polygon, list) and len(polygon) == 0:
        return False
    elif isinstance(polygon, np.ndarray) and polygon.size == 0:
        return False
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result == 1

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

def start_detection_from_file(video_path):
    global cap
    cap = cv2.VideoCapture(video_path)
    detect_objects(cap)

def detect_objects(cap):
    global count
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_detected = frame.copy()
        frame = preprocess(frame)
        results = model(frame)
        # Reset the alarm flag at the beginning of each frame
        alarm_triggered = False
        for index, row in results.pandas().xyxy[0].iterrows():
            center_x = None
            center_y = None
            if row['name'] in target_classes:
                name = str(row['name'])
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                if name == 'Violence' or name == 'weapon':
                    if inside_polygon((center_x, center_y), np.array(pts)):
                        alarm_triggered = True
            if len(pts) >= 4:
                frame_copy = frame.copy()
                cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
                frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
                cv2.polylines(frame, [np.array(pts)], True, (0, 255, 0), 2)
                cv2.putText(frame, "target", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if alarm_triggered:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def select_video():
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi;*.mkv")])
    if file_path:
        start_detection_from_file(file_path)

def close_video():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root_detection.destroy()

def show_main_window():
    root_main.deiconify()
    close_video()

def show_detection_options():
    global root_detection
    root_main.withdraw()  # Hide the main window
    root_detection = tk.Toplevel()
    root_detection.title("Detection Options")

    canvas = tk.Canvas(root_detection, width=800, height=600)
    canvas.pack()

    # Load the background image
    background_image = tk.PhotoImage(file=r"C:\FILLE\COLLEGE\S6\#mini\target-detector-yolov5\image.png")
    # Assign it to a global variable to prevent garbage collection
    root_detection.background = background_image
    # Display the image
    canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

    # Equalize button sizes and color them light grey
    btn_upload = tk.Button(root_detection, text="Upload Video", command=select_video, bg="#cccccc", width=20, height=2)
    btn_upload.place(relx=0.1, rely=0.4, anchor=tk.W, y=0)

    btn_back = tk.Button(root_detection, text="Back", command=show_main_window, bg="#cccccc", width=20, height=2)
    btn_back.place(relx=0.1, rely=0.5,anchor=tk.W, y=0)

    # Set mouse callback for drawing polygon
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", draw_polygon)

root_main = tk.Tk()
root_main.title("Object Detection")

canvas_main = tk.Canvas(root_main, width=800, height=600)
canvas_main.pack()

background_image_main = tk.PhotoImage(file=r"C:\FILLE\COLLEGE\S6\#mini\target-detector-yolov5\image.png")
canvas_main.create_image(0, 0, anchor=tk.NW, image=background_image_main)

# Equalize button sizes and color them light grey
btn_detection = tk.Button(root_main, text="Start", command=show_detection_options, bg="#cccccc", width=20, height=2)
btn_detection.place(relx=0.5, rely=0.5, anchor=tk.CENTER, y=0)

root_main.mainloop()