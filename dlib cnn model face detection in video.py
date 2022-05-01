import cv2
import dlib
def facedetection(video_source,dnn_face):
    
    while True:
        # Capture frame-by-frame
        check,frame = video_cascade.read()
        #converting to gray image for faster video processing
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect face function and it's give points value of rectnagle in the class '_dlib_pybind11.rectangles' 
        rect_face = dnnFaceDetector(gray_img, 0)
        # Loop through each face detected in the image.
        for bbox in rect_face:
            
            x = bbox.rect.left()
            y = bbox.rect.top()
            w = bbox.rect.right() - x
            h = bbox.rect.bottom() - y
            # Draw bounding box around the face on the copy of the input image using the retrieved coordinates.
            cv2.rectangle(frame,(x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video_cascade.release()
    cv2.destroyAllWindows()

#creating hob face detctor object
dnnFaceDetector = dlib.cnn_face_detection_model_v1(r"C:\Users\user\Desktop\AAmer works\mmod_human_face_detector.dat")

#video path if it is webcam then give how much web cam 0,1,2,3 or if it video path then give video path in it.
video_cascade = cv2.VideoCapture(0)

#calling function.
facedetection(video_cascade,dnnFaceDetector)

