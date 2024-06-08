import cv2
import math
import json
import logging

def load_json_as_dict(json_path):
    """
    Load the content of a JSON file into a dictionary.

    Args:
    json_path (str): The path to the JSON file.

    Returns:
    dict: The dictionary containing the JSON file content.
    """
    try:
        with open(json_path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict
    except FileNotFoundError:
        print(f"[ERROR] The file at {json_path} was not found.")
        logging.error(f"File not found: {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] The file at {json_path} is not a valid JSON file.")
        logging.error(f"Invalid JSON file: {json_path}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"[ERROR] Unexpected error: {e}")
        return {}


def init_video_capture(video_path, video_fps):
    """
    Initialize video capture from the specified video path and set the FPS.

    Args:
    video_path (str): Path to the video file.
    video_fps (int): Frames per second for the video.

    Returns:
    cv2.VideoCapture: Initialized video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, video_fps)
    return cap


def draw_lines(frame, start_line_depth, end_line_depth):
    """
    Draw horizontal lines on the frame at specified depths.

    Args:
    frame (ndarray): The current video frame.
    start_line_depth (int): Y-coordinate for the start line.
    end_line_depth (int): Y-coordinate for the end line.
    """
    height, width, _ = frame.shape
    start = (0, start_line_depth)
    end = (width, start_line_depth)
    start2 = (0, end_line_depth)
    end2 = (width, end_line_depth)
    color = (0, 255, 0)  # Green color for the lines
    thickness = 2  # Line thickness

    # Draw the lines on the frame
    cv2.line(frame, start, end, color, thickness)
    cv2.line(frame, start2, end2, color, thickness)


def detect_cars(frame, car_classifier):
    """
    Detect cars in the given frame using the specified classifier.

    Args:
    frame (ndarray): The current video frame.
    car_classifier (cv2.CascadeClassifier): The classifier used for car detection.

    Returns:
    list: List of bounding boxes for detected cars.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    return cars


def draw_rectangles_and_points(frame, cars, start_line_depth, end_line_depth, frame_number, frame_skip):
    """
    Draw rectangles around detected cars and points at their centers.
    Increment count if a car passes between the specified lines.

    Args:
    frame (ndarray): The current video frame.
    cars (list): List of bounding boxes for detected cars.
    start_line_depth (int): Y-coordinate for the start line.
    end_line_depth (int): Y-coordinate for the end line.
    frame_number (int): The current frame number.
    frame_skip (int): Number of frames to skip before counting a car.

    Returns:
    tuple: List of considered coordinates and the count increment.
    """
    considered_coordinates = []
    count_increment = 0

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw rectangle around detected car
        coordinate = (round(x + w / 2), round(y + w / 2))  # Calculate center coordinate of the car

        # Check if the car is within the specified lines and if the frame number is a multiple of frame_skip
        if start_line_depth <= coordinate[1] <= end_line_depth and frame_number % frame_skip == 0:
            count_increment += 1
            considered_coordinates.append((x, y, w, h))

        point_color = (0, 0, 255)  # Red color for the point
        point_thickness = 5  # Point thickness
        cv2.circle(frame, coordinate, point_thickness, point_color, -1)  # Draw point at the center of the car

    return considered_coordinates, count_increment


def count_nearby_coordinates(set1, set2, max_separation=17):
    """
    Count the number of coordinates in set1 that are near any coordinate in set2.

    Args:
    set1 (list): List of coordinates from the previous frame.
    set2 (list): List of coordinates from the current frame.
    max_separation (int): Maximum distance to consider two coordinates as nearby.

    Returns:
    int: Number of nearby coordinates.
    """
    count = 0
    for (x1, y1, w1, h1) in set1:
        for (x2, y2, w2, h2) in set2:
            distance = math.sqrt(((x2 + w2 / 2) - (x1 + w1 / 2)) ** 2 + ((y2 + h2 / 2) - (y1 + h1 / 2)) ** 2)
            if distance <= max_separation:
                count += 1
                break
    return count


def draw_text(frame, text, position=(50, 50)):
    """
    Draw text on the frame at the specified position.

    Args:
    frame (ndarray): The current video frame.
    text (str): The text to be drawn.
    position (tuple): The (x, y) position to place the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    color = (255, 0, 0)  # Blue color for the text
    thickness = 2  # Text thickness
    cv2.putText(frame, text, position, font, font_size, color, thickness)

def main(video_path, car_classifier, start_line_depth, end_line_depth, frame_skip, centroid_separation, video_fps):
    """
    Main function to process the video, detect cars, and count them.

    Args:
    video_path (str): Path to the video file.
    car_classifier (cv2.CascadeClassifier): The classifier used for car detection.
    start_line_depth (int): Y-coordinate for the start line.
    end_line_depth (int): Y-coordinate for the end line.
    frame_skip (int): Number of frames to skip before counting a car.
    centroid_separation (int): Maximum distance to consider two coordinates as nearby.
    video_fps (int): Frames per second for the video.
    """
    cap = init_video_capture(video_path, video_fps)
    count = 0
    frame_number = 0
    previous_coordinates = []

    while True:
        frame_number += 1
        ret, frame = cap.read()
        if not ret:
            break

        draw_lines(frame, start_line_depth, end_line_depth)  # Draw the lines on the frame
        cars = detect_cars(frame, car_classifier)  # Detect cars in the current frame
        considered_coordinates, count_increment = draw_rectangles_and_points(
            frame, cars, start_line_depth, end_line_depth, frame_number, frame_skip
        )

        count += count_increment
        count -= count_nearby_coordinates(previous_coordinates, considered_coordinates, max_separation=centroid_separation)
        draw_text(frame, f"{count}")  # Draw the count on the frame

        # Log the count with a timestamp
        logging.info(f"[INFO] Frame {frame_number}: Car count is {count}")
        
        previous_coordinates = considered_coordinates  # Update the previous coordinates
        cv2.imshow('frame', frame)  # Display the frame
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
