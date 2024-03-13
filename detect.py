import os
import cv2
from ultralytics import YOLO

model = YOLO("best_M_model.pt")

video_path = 'rtsp://admin:123456789a@10.113.100.97:554/1'  # Change this to your video file's path

cap = cv2.VideoCapture(video_path)
output_filename = "result"

if not os.path.exists(output_filename):
    os.makedirs(output_filename)

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0]) * 100
            model  = box.cls
            print("Detected: ", result.names[int(box.cls[0])])
            print("Conf: ", conf)
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])] , int(conf)} % ",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

writer = create_video_writer(cap, output_filename)
try:
    while True:

        success, img = cap.read()

        if not success:
            break

        result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
        
        writer.write(result_img)
        cv2.imshow("Image", result_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print("Error occured: ", e)

finally:
    cap.release()
    if writer:
        writer.release()

    cv2.destroyAllWindows()
