import logging
from datetime import datetime

from utils import *

# Set up logging
logging.basicConfig(filename='logs/car_count.log', level=logging.INFO, format='%(asctime)s - %(message)s')


if __name__ == "__main__":

    # Load the input parameters from JSON
    jsonData = load_json_as_dict("config.json")
    # Video Path
    videoPath = jsonData["videoPath"]
    # Video Configuration
    videoFPS = jsonData["videoFPS"]
    # Inference Configuration
    frameSkip = jsonData["frameSkip"]
    car_classifier = cv2.CascadeClassifier(jsonData["modelPath"])
    startLineDepth = jsonData["startROILineDepth"]
    endLineDepth = jsonData["endROILineDepth"]
    centroidSeparation = jsonData["centroidSeparation"]

    # Call the main function with the defined parameters
    main(videoPath, car_classifier, startLineDepth, endLineDepth, frameSkip, centroidSeparation, videoFPS)
