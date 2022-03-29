import io
import os
from enum import Enum
from tempfile import mkdtemp

import cv2
import cvlib as cv
import numpy as np
import uvicorn
from cvlib.object_detection import draw_bbox
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Deploying a ML Model with FastAPI")

TEMP_OUTPUT_DIR = mkdtemp()


class Model(str, Enum):
    """
    List available models using Enum for convenience.
    """
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


@app.post("/predict")
def prediction(model: Model, image_file: UploadFile = File(...)):
    """
    Endpoint to handles the object detection request.

    :param model: either yolov3-tiny or yolov3 enum.
    :param image_file: the file reference to the target image

    """

    file_extension = image_file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(
            status_code=415, detail="Unsupported file provided."
        )
    image = image2cv2(image_file)
    bbox, label, conf = cv.detect_common_objects(image, model=model)
    output_image = draw_bbox(image, bbox, label, conf)
    temp_file = f"{TEMP_OUTPUT_DIR}/{image_file.filename}"
    if not cv2.imwrite(temp_file, output_image):
        raise Exception("Could not save temporary image in the server")

    tagged_image = open(temp_file, mode="rb")

    return StreamingResponse(tagged_image, media_type="image/jpeg")


def image2cv2(image_file: UploadFile = File(...)):
    """
    Decode the image file

    This function reads the image as a stream of bytes,
    writes its stream of bytes into a numpy array, and
    decode it as an image.
    Args:
        image_file: the file reference to the image

    Returns:
        the CV2 image object

    """
    image_stream = io.BytesIO(image_file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


if __name__ == "__main__":
    # Host depends on the setup you selected (docker or virtual env)
    host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
    uvicorn.run(app, host=host, port=8000)
