# face-detection
This is a ml model that is trained to detect human face and display the person's name. The code is written in python using the face recognition library and OpenCV. By comparing face encodings of known faces with those detected in a test image, the program identifies individuals and displays their names on the image. This approach involves loading images from a training directory, computing face encodings, and storing them with corresponding names. The system then processes a test image, detects faces, compares their encodings, and determines the best match. Finally, it draws rectangles around the recognized faces and displays the names, providing a practical application of face recognition technology.
