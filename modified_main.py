from google.colab import drive
drive.mount('/content/drive')
! pip install face_recognition
import face_recognition as fr
import cv2
import numpy as np
import os
import sys
from google.colab.patches import cv2_imshow
def trainpro(known_names,known_name_encodings):
 path = "/content/drive/MyDrive/face-recognition-python-code/train"
# to be changed into a dictionary from a text path
 images = os.listdir(path)
 for _ in images:
    image = fr.load_image_file(path + "/"+_)
    image_path = path + "/"+_
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

 print(known_names)
 def facerec(test_image,known_names,known_name_encodings):

  image = cv2.imread(test_image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  face_locations = fr.face_locations(image)
  face_encodings = fr.face_encodings(image, face_locations)

  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)

    if matches[best_match]:
        name = known_names[best_match]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
  cv2_imshow(image)
  cv2.imwrite("./output.jpg", image)
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
  from IPython.display import Image
test_image=""
known_names = []
known_name_encodings = []
trainpro(known_names,known_name_encodings)
print("Do you want to give access to camera ?")
ans=input()
if(ans=="y"):
 try:
  test_image = take_photo()
  facerec(test_image,known_names,known_name_encodings)
 except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
else:
 test_image = "/content/drive/MyDrive/face-recognition-python-code/test/test.jpg"
 facerec(test_image,known_names,known_name_encodings)

cv2.waitKey(0)
cv2.destroyAllWindows()
