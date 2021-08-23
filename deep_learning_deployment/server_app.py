import io
import os
from enum import Enum

import cv2
import cvlib as cv
import numpy as np
import uvicorn
from cvlib.object_detection import draw_bbox
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Deploying a ML Model with FastAPI")


class Model(str, Enum):
    """
    List available models using Enum for convenience.

    This is useful when the options are pre-defined.
    """

    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    """
    This endpoint handles all the logic necessary for the object detection to work.

    It requires the desired model and the image in which to perform object detection.
    """
    # 1. VALIDATE INPUT FILE
    filename = file.filename
    file_extension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(
            status_code=415, detail="Unsupported file provided."
        )

    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    image = image2cv2(file)

    # 3. RUN OBJECT DETECTION MODEL

    # Run object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    # Create image that includes bounding boxes and labels
    output_image = draw_bbox(image, bbox, label, conf)

    # Save it in a folder within the server
    cv2.imwrite(f"images_uploaded/{filename}", output_image)

    # 4. STREAM THE RESPONSE BACK TO THE CLIENT

    # Open the saved image for reading in binary mode
    file_image = open(f"images_uploaded/{filename}", mode="rb")

    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")


def image2cv2(file):
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)

    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


if __name__ == "__main__":
    # Host depends on the setup you selected (docker or virtual env)
    host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
    # Spin up the server!
    uvicorn.run(app, host=host, port=8000)
