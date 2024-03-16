import os
import cv2
from ultralytics import YOLO
import numpy as np
import time
import yaml
import socket
import json
import threading
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
object_detected_time = None

actual_model = "None"
cap = cv2.VideoCapture(video_path)

Yolo_model = YOLO(Yolo_model_name)

samsung_rb = []
samsung_rt = []
artel = []
shivaki = []
berg = []
maunfeld = []
bar_codes_path = "bar_codes.json"
def get_barcode() -> None:
    global samsung_rb, samsung_rt, artel, berg, shivaki, maunfeld

    with open(bar_codes_path, 'r') as file:
        data = json.load(file)
    samsung_rb = data["Samsung RB"]
    samsung_rt = data["Samsung RT"]
    artel = data["Artel"]
    shivaki = data["Shivaki"]
    berg = data["Berg"]
    maunfeld = data["Maunfeld"]

def check_barcode(barcode:str) -> str:
    model = barcode[:4]
    if model in samsung_rt:
        return "Samsung RT"
    elif model in samsung_rb:
        return "Samsung RB"
    elif model in artel:
        return "Artel"
    elif model in shivaki:
        return "Shivaki"
    elif model in berg:
        return "Berg"
    elif model in maunfeld:
        return "Maunfeld"
    else:
        return "Not Specified Barcode"

def start_tcp_server() -> None:
    global actual_model
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
                    decoded_data = data.decode('utf-8').replace('\r\n', '')
                    print(f"Received data: {decoded_data}")
                    actual_model = check_barcode(decoded_data)
                    print(f"Fridge Model: {actual_model}")

            except KeyboardInterrupt:
                print('Socket close')
                client_socket.close()
                break

            except Exception as e:
                print(f"Error while handling client connection: {e}")

            finally:
                client_socket.close()

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        server_socket.close()

def set_full_screen_mode(frame_name):
    cv2.namedWindow(frame_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def predict(chosen_model, img, classes=[], conf=confidance):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=confidance)
    else:
        results = chosen_model.predict(img, conf=confidance)

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

def connect_camera(video_path):
    print("Connecting...")
    while True:
        try:
            if cam_force_address is not None:
                requests.get(cam_force_address)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                time.sleep(reconnection_time)
                raise Exception("Could not connect to a camera: {0}".format(video_path))

            if cap.isOpened():
                print("Connected to a camera: {}".format(video_path), flush=True)
                break

        except Exception as e:
            print(e)

            if blocking is False:
                break

            time.sleep(reconnection_time)

    return cap


def mask_frame(image, line_color, rect_width, rect_height):
    # Calculate the position of the rectangle
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2
    top_left = (x_center - rect_width // 2, y_center - rect_height // 2)
    bottom_right = (x_center + rect_width // 2, y_center + rect_height // 2)
    cv2.rectangle(image, top_left, bottom_right, line_color, thickness=8)

    # Create a mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

    # Blur the entire image
    blurred_image = cv2.GaussianBlur(image, (41, 41), 0)

    # Apply the mask to combine the blurred image and the original image
    masked = np.where(mask[:, :, None].astype(bool), image, blurred_image)
    return masked

def put_text_on_frame(frame, model):
    x_center = frame.shape[1] // 2
    text_size = cv2.getTextSize(model, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
    w = int(x_center - text_size[0]/2)
    cv2.putText(frame, model, (w, 100), cv2.FONT_HERSHEY_SIMPLEX, 4,(0, 255, 0), 6)

def detection(result_img, detected_model, actual_model):
    global object_detected_time
    if detected_model == actual_model:  # Replace with your object's label
        if object_detected_time is None:
            object_detected_time = time.time()
        elif time.time() - object_detected_time >= 1:
            # Draw a tick on the frame
            cv2.circle(result_img, (200, 200), 30, (0, 255, 0), thickness=5)
        line_color = (0,255,0)

    elif detected_model != 'None':
        line_color = (0,0,255)
        cv2.drawMarker(result_img, (200, 200),(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=50, thickness=5)

    else:
        object_detected_time = None
        line_color = (255,0,0)

    return line_color

def remap_detected_model(detected_model):
    if detected_model[:5] == 'Artel':
        detected_model = 'Artel'

    elif detected_model[:11] == 'Samsung_RB':
        detected_model = 'Samsung RB'

    elif detected_model[:11] == 'Samsung_RT':
        detected_model = 'Samsung RT'

    elif detected_model[:7] == 'Shivaki':
        detected_model = 'Shivaki'

    return detected_model

if __name__ == "__main__":
    tcp_server_thread = threading.Thread(target=start_tcp_server)
    tcp_server_thread.start()
    try:
        set_full_screen_mode(screen_frame_name)
        get_barcode()
        while True:
            success, img = cap.read()

            if not success:
                print("Stream stops!", flush=True)
                cap = connect_camera(video_path)
                continue

            result_img, detected_model = predict_and_detect(Yolo_model, img, classes=[], conf=confidance)
            detected_obj = remap_detected_model(detected_model)

            line_color = detection(result_img, detected_obj,actual_model)
            mask = mask_frame(result_img, line_color, rect_width, rect_height)
            put_text_on_frame(mask, actual_model)

            cv2.imshow(screen_frame_name, mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print("Error occured: ", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
