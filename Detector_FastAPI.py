import cv2
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

app = FastAPI()

@app.post("/imagen/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):

    extension = uploaded_file.filename.split('.')[-1]
    file_ = tempfile.NamedTemporaryFile(suffix='.' + extension)
    file_.write(uploaded_file.file.read())

    plate_weights = "placas2.weights"
    ch_weights = "cv2.weights"
    config = "tiny-yolo.cfg"
    up_model = "Nadam45.model"

    result = []

    image = cv2.imread(file_.name)
    height, width, _ = image.shape

    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416),
        (0,0,0), True, crop = False)
    
    plate_net = cv2.dnn.readNet(plate_weights, config)
    ch_net = cv2.dnn.readNet(ch_weights, config)

    plate_net.setInput(blob)
    plate_net_outputs = plate_net.forward(get_output_layers(plate_net))

    plates = []

    for plate_detection in plate_net_outputs:
        for plate_coord in plate_detection:
            if plate_coord[5] > 0.5:
                center_x = int(plate_coord[0]*width)
                center_y = int(plate_coord[1]*height)
                w = int (plate_coord[2]*width)
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

    plates.sort(reverse = True)
            
    model = load_model(up_model)
    Characters = []
    coordts = []

    abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z"]
    
    numbers = []
    letters = []

    plate = plates[0][1]

    height, width, _ = plate.shape

    blob2 = cv2.dnn.blobFromImage(plate, scale, (416, 416),
        (0,0,0), True, crop = False)

    ch_net.setInput(blob2)
    ch_net_outputs = ch_net.forward(get_output_layers(ch_net))

    for ch_detection in ch_net_outputs:
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

    for ch in Characters:

        ch = ch[1]
        ch = cv2.resize(ch, (24, 24))
        ch = ch/255
        ch = cv2.dnn.blobFromImage(np.float32(ch), 1.0,
            (24, 24), (0,0,0), False, crop = False)
            
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

    
    return {
        "Placa": result,
    }