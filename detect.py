import os
import cv2
from ultralytics import YOLO
import numpy as np
import time
import yaml
import socket
import json
import threading
import multiprocessing
from typing import Tuple
# from checkbarcode import BarCodeCheck
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

Yolo_model_name = config['model_name']
confidance = config['confidance']
video_path = config['stream_url']
reconnection_time = config["reconnection_time"]
screen_frame_name = config["screen_frame_name"]
rect_width = config['rect_width']
rect_height = config['rect_height']
host = config['tcp_server_ip']
port = config['tcp_server_port']
bar_codes_path = config['barcodes_file']
object_detected_time = None

barcode = 'None'
brand = 'None'
model_name = 'None'
model_color = ( 0, 0, 0)
cam_force_address = None
Yolo_model = YOLO(Yolo_model_name)

samsung_rb = {}
samsung_rt = {}
artel = {}
shivaki = {}
berg = {}
maunfeld = {}
global ok_img
ok_img = cv2.imread('ok.png')
def get_models() -> None:
    global samsung_rb, samsung_rt, artel, berg, shivaki, maunfeld

    with open(bar_codes_path, 'r') as file:
        data = json.load(file)
    samsung_rb = data["Samsung RB"]
    samsung_rt = data["Samsung RT"]
    artel = data["Artel"]
    shivaki = data["Shivaki"]
    berg = data["Berg"]
    maunfeld = data["Maunfeld"]

def check_barcode(barcode:str) -> Tuple[str, str]:
    model = barcode[:4]
    global model_color
    if len(model) >= 4:
        if model in samsung_rt.keys():
            model_color = (255, 0, 0)
            return ("Samsung RT", samsung_rt[model])
        elif model in samsung_rb.keys():
            model_color = (255, 0, 0)
            return ("Samsung RB", samsung_rb[model])
        elif model in artel.keys():
            model_color = (0, 255, 0)
            return ("Artel", artel[model])
        elif model in shivaki.keys():
            model_color = (0, 0, 255)
            return ("Shivaki", shivaki[model])
        elif model in berg.keys():
            model_color = (0, 255, 255)
            return ("Berg", berg[model])
        elif model in maunfeld.keys():
            model_color = (128, 128, 128)
            return ("Maunfeld", maunfeld[model])
        else:
            return ("None", "None")
    else:
        return ("None", "None")

def start_tcp_server() -> None:
    global brand, barcode, model_name
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host,port))

        server_socket.listen(5)

        print(f"TCP server is listening on {host}:{port}")

        while True:
            client_socket, client_address = server_socket.accept()

            print(f"Accepted connection from {client_address}")

            try:
                data = client_socket.recv(1024)

                if data:
                    barcode = data.decode('utf-8').replace('\r\n', '')
                    print(f"Received data: {barcode}")
                    brand, model_name = check_barcode(barcode)
                    print(f"Fridge Brand: {brand}  Model: {model_name}")

            except Exception as e:
                print(f"Error while handling client connection: {e}")

            finally:
                client_socket.close()
    except KeyboardInterrupt:
        print('Server Socket close')
        server_socket.close()

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        server_socket.close()

def set_full_screen_mode(frame_name):
    cv2.namedWindow(frame_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def predict(chosen_model, img, classes=[], conf=confidance):
    if classes:
        results = chosen_model.predict(img, classes=classes, verbose=False, conf=confidance)
    else:
        results = chosen_model.predict(img, verbose=False, conf=confidance)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=confidance):
    model = "None"
    results = predict(chosen_model, img, classes, conf=confidance)
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0]) * 100
            model  = result.names[int(box.cls[0])]
            print("Detected: ", result.names[int(box.cls[0])])
            print("Conf: ", conf)
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])] , int(conf)}%",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, model

def connect_to_camera(video_path):
  """Connects to the RTPS camera and returns the capture object."""
  while True:
    try:
      cap = cv2.VideoCapture(video_path)
      if cap.isOpened():
        return cap
      else:
        print("Failed to open camera. Retrying...")
    except Exception as e:
      print(f"Error connecting to camera: {e}")
    finally:
      # Wait 10 seconds before retrying
      cv2.waitKey(10000)


def mask_frame(image, model_color, rect_width, rect_height):
    # Calculate the position of the rectangle
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2
    border_color = model_color  # Choose the color of the border (in BGR format)
    bordered_frame = cv2.copyMakeBorder(image, 50, 50, 50 , 50, cv2.BORDER_CONSTANT, value=border_color)

    return bordered_frame

def put_text_on_frame(frame, model):
    x_center = frame.shape[1] // 2
    text_size = cv2.getTextSize(model, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    w = int(x_center - text_size[0]/2)
    cv2.putText(frame, model, (w, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, barcode, (frame.shape[1] - 400, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

def detection(result_img, detected_model, actual_model):
    global object_detected_time
    if actual_model == "Maunfeld" or actual_model == "Berg":
        actual_model = "Artel"

    if detected_model == actual_model:  # Replace with your object's label
        # if object_detected_time is None:
        #     object_detected_time = time.time()
        # elif time.time() - object_detected_time >= 1:
            # Draw a tick on the frame
        cv2.putText(result_img, "OK", (10, result_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)


    elif detected_model != 'None':
        cv2.putText(result_img, "NG", (10, result_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        object_detected_time = None



def remap_detected_model(detected_model):
    if detected_model[:5] == 'Artel' :
        detected_model = 'Artel'
    elif detected_model[:10] == 'Samsung_RT':
        detected_model = 'Samsung RT'
    elif detected_model[:10] == 'Samsung_RB':
        detected_model = 'Samsung RB'
    elif detected_model[:7] == 'Shivaki':
        detected_model = 'Shivaki'

    return detected_model

if __name__ == "__main__":
    get_models()
    tcp_server_thread = threading.Thread(target=start_tcp_server)
    tcp_server_thread.start()
    cap = connect_to_camera(video_path)
    try:

        set_full_screen_mode(screen_frame_name)
        while True:
            success, img = cap.read()

            if not success:
                print("Stream stops!", flush=True)
                cap.release()
                cap = connect_to_camera(video_path)
                continue

            result_img, detected_model = predict_and_detect(Yolo_model, img, classes=[], conf=confidance)
            detected_obj = remap_detected_model(detected_model)
            detection(result_img, detected_obj,brand)
            mask = mask_frame(result_img, model_color, rect_width, rect_height)
            put_text_on_frame(mask, brand + " " + model_name)

            cv2.imshow(screen_frame_name, mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error occured: ", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
