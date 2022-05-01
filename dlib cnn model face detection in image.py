import cv2
import dlib
def face_detection(image,model):
        #converting to gray image for faster processing
        grey_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        #detect face function and it's give points value of rectnagle in the class '_dlib_pybind11.rectangles'         
        results = dnnFaceDetector(grey_image, 0)
        print("Number of faces detected: {}".format(len(results)))
        print(type(results))
        print(results)
        # Loop through each face detected in the image.
        for bbox in results:
                x = bbox.rect.left()
                y = bbox.rect.top()
                w = bbox.rect.right() - x
                h = bbox.rect.bottom() - y
                print(x,y,w,h)
                # Draw bounding box around the face on the copy of the input image using the retrieved coordinates.
                cv2.rectangle(input_image,(x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
        image_resize = cv2.resize(input_image,(int(input_image.shape[1]/2),int(input_image.shape[0]/2)))
        cv2.imshow('Detected faces', image_resize) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

dnnFaceDetector = dlib.cnn_face_detection_model_v1(r"C:\Users\user\Desktop\AAmer works\mmod_human_face_detector.dat")
input_image = cv2.imread(r"C:\Users\user\Desktop\AAmer works\762063.1.png")
face_detection(input_image,dnnFaceDetector)
