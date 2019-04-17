This repository contains the code and all the pre trained files which are used for liveliness detection of human face for building an anti spoofing algorithm. Here we capture the eye-blink movement using EAR(Eye Aspect Ratio) to capture blinking of eyes. Any still image will have a constant EAR value while for a real face EAR will keep on changing and will drop as the person will blink the eye. 


Download dat file of shape predictor from the following link.
https://drive.google.com/open?id=1zw4akec8LC-PurvaeH0ergwJJ1XQgX_6

Download all the files and save it in a directory.
Now Open your command prompt and move to the directory where all the files have been stored.
Run the following command on your prompt.

python major10.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel --shape-predictor shape_predictor_68_face_landmarks.dat




