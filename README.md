# Computer-Vision-Based-Mouse
 A computer vision based mouse, which can control and command the cursor of a computer or a computerized system using a  Web camera. In order to move the cursor on the computer screen the user simply uses an Orange Color within the viewing area of the camera. The video generated by the camera is analyzed using computer vision techniques and the computer moves the cursor according to color movements. The computer vision based mouse also have a feature of Right-click and Left-click.To  left-click,I have used a color green and for Right-click,I have used a Yellow color.
 
In order to implement this program we need to have following things installed in our system:

 1. Python3
 2. pip
 3. Opencv
 4. Matplotlib
 5. Numpy
 6. pyautogui
 
### To install opencv:
```
python -m pip install -U pip opencv -python
```
 
We have implemented this program with Python3 . Opencv is used for Computer Vision. Every image or video captured by opencv is done with OpenCV. PyautoGui is used for mouse activities. So movement of cursor, its click event functions is because of pyautogui.


# Video Analysis  And Recognisation System
The video transmitted by the camera is analyzed image by
image in real time.  The first method is based on edge detection. In this
method, the edges of the mouse are detected by an edge
detection method. In this method the the Orange color is taken to detect an 
edge of a mouse. Any object of orange color whch lies at arange of (5 to 15 HSV)
can be taken. When the web camera started every object aside from orange color
wll be black. Only object with orange color will be available.

   The second method for mouse detection and tracking
method that we describe is based on color analysis. In this
approach an object with orange color is placed in front of the camera.This
is known as reference mark.
This mark is used as the reference point
of the mouse. Whenever the object is moved by the user
the location of the reference point changes. The detection
of the reference point can be carried out by adaptive
thresholding. By tracking the reference point the cursor is
moved by the computer as described above. This color
coding approach is easier and more robust then the edge
detection method as the viewing area of the camera is
more or less known by the image analysis system


The third detection and tracking approach is based on
motion analysis. WE will achieve it through by using the above 2 methods
with pyautogui to move the mouse based on its colors.

Orange - Cursor movement

Green - Left-Click

Yellow - Rght-click
 
### To see the demo click https://www.youtube.com/watch?v=35sj52cKVDg
### To read this project's publication  http://ijcsmc.com/docs/papers/May2020/V9I5202006.pdf
