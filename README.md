# Driver-Drowsiness-Detection
It is our Minor Research Project in B.tech 6th semester.

Road accidents resulting in deaths and several injuries are due to drowsiness or lack of proper sleep. To control such accidents from happening, we need to take care of drivers, who are driving while they feel drowsy. To deal with this problem we implement a computer vision base approach that detects drowsiness of driver based on his facial features. We implement an eye blink and yawning monitoring system that uses eye feature points to determine the open or closed state of the eye and mouth feature points to determine the state of the mouth. It activates an alarm if the driver feels drowsy. So that accidents do not happen due to their lack of control on vehicle because of drowsiness. We are using a web-camera that captures video frames continuously. After processing that frames, we get face from that captured frame images and apply landmarks on it. We extract region of eyes and mouth with coordinates from that landmarks of the face. After getting eyes and mouth region, we are counting the number of blinks and yawning frames. It is marked that when people feel drowsy then blink and yawn ratio increased rapidly. According to that, we count the number of blinks and yawn, and fetch number of video frames in which the blinking and yawning rate of the driver is continuously increasing. If the number of frames containing blink or yawn increases above a particular count than we can say that driver is feeling drowsy while driving. When we detect such a state of the driver then our system alerts the driver via alarm. So, the total number of accidents due to drowsiness can be decreased.


The Existing Systems includes computer vision based solutions as well as hardware based solutions which include some sensors and devices like Electrocardiogram, Electroencephalogram, etc. Electrocardiogram is used to recognition and record any electrical activity within the heart. Electroencephalogram is used to record electrical activity of brain. Some systems used pressure sensor to measure hand gripping pressure. This hardware based systems are costly.

We focused on computer vision based approaches that uses only camera to detect drowsiness.Our Drowsiness Detection System consist of face detection, facial landmark detection, extraction of region of interest, calculating mean values and finding distance.  

<h2>System Flow:</h2>

Capturing Video and Color Conversion

Face Detection in Captured Video Frames

Detect the Key Facial Structures on the Face

Extract the Region of Interest from Facial Landmarks

Finding Mean and Distance

Detecting Drowsiness

<h2>Feasibility Study</h2>

Proper Lighting Condition

Proper Camera Angle


<h2>Hardware Requirements:</h2>

The following describes the hardware needed in order to execute and develop the Driver Drowsiness detection application:
<h4>Tested on Laptop with hardware configuration:</h4>

RAM: 8 GB

OS: Windows 10 Home

Processor: intel-core i5-8250U

Camera type: In-Built Camera

System bit: 64 bits

The Computer Desktop or Laptop will be utilized to run visual software in order to display what camera has captured.

<h4>Web camera or In-built Camera within PC:</h4>

Webcam or in-built camera of laptop is utilized for image processing techniques. The webcam will continuously take image in order to process the image and find pixel position. Camera feed is used to interact with the digital world content with the concept of human computer interaction.

<h4>Hoping to test in future on:</h4>

Raspberry PI 3 model B with PI camera

<h2>Software Requirements:</h2>
 
Python  3.6.1

OpenCV-python  4.1.1.26

SciPy  1.4.1

Imutils  0.5.3

Playsound  1.2.2

Dlib  19.16.0


