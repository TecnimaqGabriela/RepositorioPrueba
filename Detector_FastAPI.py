import tempfile
import numpy as np

import cv2
from fastapi import FastAPI, File, UploadFile, Form
from tensorflow.keras.models import load_model

app = FastAPI(version='0.1.1')

@app.post("/imagen/{placa}")
async def create_upload_file(placa: str, uploaded_file: UploadFile = File(...)):
    given_plate = placa.upper()
    extension = uploaded_file.filename.split('.')[-1]
    file_ = tempfile.NamedTemporaryFile(suffix='.' + extension)
    file_.write(uploaded_file.file.read())

    image = cv2.imread(file_.name)
    height, width, _ = image.shape

    plate_weights = "placas2.weights"
    config = "tiny-yolo.cfg"

    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416),
        (0,0,0), True, crop = False)

    plate_net = cv2.dnn.readNet(plate_weights, config)
    plate_net.setInput(blob)
    plate_layers = plate_net.getLayerNames()
    plate_output = [plate_layers[i[0] - 1] for i in plate_net.getUnconnectedOutLayers()]
    plate_outputs = plate_net.forward(plate_output)

    plates = []

    for plate_detection in plate_outputs:
        for plate_coord in plate_detection:
            print(plate_coord)
            if plate_coord[5] > 0.5:
                center_x = int(plate_coord[0]*width)
                center_y = int(plate_coord[1]*height)
                w = int(plate_coord[2]*width)
                h = int(plate_coord[3]*height)
                x = int(center_x-(w/2))
                y = int(center_y-(h/2))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                plate = image[y:y+h, x:x+w]
                plate_with_area = (h*w, plate)
                plates.append(plate_with_area)

    ch_weights = "ch2.weights"
            
    Characters = []

    abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z"]
    
    numbers = []
    letters = []
    result = []

    plates.sort(reverse = True)
    plate = plates[0][1]

    height, width, _ = plate.shape

    blob2 = cv2.dnn.blobFromImage(plate, scale, (416, 416),
        (0,0,0), True, crop = False)

    ch_net = cv2.dnn.readNet(ch_weights, config)
    ch_net.setInput(blob2)
    ch_layers = ch_net.getLayerNames()
    ch_output = [ch_layers[i[0] - 1] for i in ch_net.getUnconnectedOutLayers()]
    ch_outputs = ch_net.forward(ch_output)

    for ch_detection in ch_outputs:
        for ch_coord in ch_detection:
            if ch_coord[5] > 0.5:
                center_x = int(ch_coord[0]*width)
                center_y = int(ch_coord[1]*height)
                w = int(ch_coord[2]*width)
                h = int(ch_coord[3]*height)
                x = int(center_x-(w/2))
                y = int(center_y-(h/2))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                ch = plate[y:y+h, x:x+w]
                coord = (x, ch)
                Characters.append(coord)
            
    Characters.sort()

    model = load_model("Nadam45.model")

    for ch in Characters:

        ch = ch[1]
        ch = cv2.resize(ch, (24, 24))
        ch = ch/255
        ch = cv2.dnn.blobFromImage(np.float32(ch), 1.0, (24, 24),
            (0,0,0), False, crop = False)

        prediction = model.predict(ch)

        if np.argmax(prediction) < 10:
            result.append(np.argmax(prediction))
        else:
            result.append(abc[np.argmax(prediction) - 10])

        possibl_numbers = prediction[0][:10]
        possibl_letters = prediction[0][10:]

        numbers.append(np.argmax(possibl_numbers))
        letters.append(abc[np.argmax(possibl_letters)])

    if len(result) == 6:
        if type(result[0]) == np.int64 and type(result[1]) == str and type(result[2]) == str and type(result[3]) == np.int64 and type(result[4]) == np.int64:
            result[0] = letters[0]
        elif type(result[1]) == np.int64 and type(result[0]) == str and type(result[2]) == str and type(result[3]) == np.int64 and type(result[4]) == np.int64:
            result[1] = letters[1]
        elif type(result[2]) == np.int64 and type(result[1]) == str and type(result[0]) == str and type(result[3]) == np.int64 and type(result[4]) == np.int64:
            result[2] = letters[2]
        elif type(result[0]) == str and type(result[1]) == str and type(result[2]) == str and type(result[3]) == str and type(result[4]) == np.int64:
            result[3] = numbers[3]
        elif type(result[0]) == str and type(result[1]) == str and type(result[2]) == str and type(result[3]) == np.int64 and type(result[4]) == str:
            result[4] = numbers[4]
        
        if type(result[0]) == np.int64 and type(result[1]) == np.int64 and type(result[2]) == str and type(reuslt[3]) == np.int64 and type(result[4]) == np.int64:
            result[0] = letters[0]
            result[1] = letters[1]
        if type(result[0]) == np.int64 and type(result[1]) == str and type(result[2]) == np.int64 and type(result[3]) == np.int64 and type(result[4]) == np.int64:
            result[0] = letters[0]
            result[2] = letters[2]
        if type(result[0]) == str and type(result[1]) == np.int64 and type(result[2]) == np.int64 and type(result[3]) == np.int64 and type(result[4]) == np.int64:
            result[1] = letters[1]
            result[2] = letters[2]
        if type(result[0]) == str and type(result[1]) == str and type(result[2]) == str and type(result[3]) == str and type(result[4]) == str:
            result[3] = numbers[3]
            result[4] = numbers[4]

        if type(result[0]) == str and type(result[1]) == np.int64 and type(result[2]) == str and type(result[3]) == np.int and type(result[4]) == str:
            result[1] = letters[1]
            result[3] = numbers[3]
        if type(result[0]) == np.int64 and type(result[1]) == str and type(result[2]) == np.int64 and type(result[3]) == str and type(result[4]) == np.int64:
            result[0] = letters[0]
            result[2] = letters[2]
            result[3] = numbers[3]
        if type(result[0]) == np.int64 and type(result[1]) == str and type(result[2]) == np.int64 and type(result[3]) == np.int64 and type(result[4]) == str:  
            result[0] = letters[0]
            result[2] = letters[2]
            result[4] = numbers[4]
        
    if len(result) == 5:

        if type(result[0]) == str and type(result[1]) == np.int64 and type(result[2]) == np.int64 and type(result[3]) == np.int64:
            result[1] == letters[1]
        if type(result[0]) == np.int64 and type(result[1]) == str and type(result[2]) == np.int64 and type(result[3]) == np.int64:
            result[0] = letters[0]

    Coincidencias = []

    for i in range(len(result)):
        if str(result[i]) == str(given_plate[i]):
            Coincidencias.append(True)
        else:
            Coincidencias.append(False)
        
    if all(Coincidencias):
        end = {"Message": "Las placas coinciden"}
    else:
        end = {
            "Message": "Las placas no coinciden",
            "Placa dada por el usuario": given_plate,
            "Placa leída en la imágen": str(result)
        }

    return end
