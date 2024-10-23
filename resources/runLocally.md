## To run the application in your local machine:

- ### <b>pull</b> a standalone snapshop of the app that can run on any machine:

    - > `docker pull dockterduck/fingertips:latest`

    <br>

- ### verify the image locally

    - > `docker images`

<br> 

    expect something like this:


| REPOSITORY    |   TAG   |    IMAGE ID  |   CREATED  |   SIZE |
| --- | --- | --- | --- | --- |
| dockterduck/fingertips |  latest  |  c10f130bcbef |  5 minutes ago |  1.35GB |


<br>

- ### run the application as a background process:
    
    - > `docker run -d -p 5000:5000 --name fingertips fingertips_img `

    
    [flask runs on port `5000` by default, we map 5000 port of localhost i.e. 127.0.0.1:5000  to the internal port 5000 of the container using the <b>'-p'</b> flag]


<br>

- ### open and test the application on localhost:5000

<br>

<hr>


### NOTE: On Virtual Machines (VMs) or Windows Subsystem for Linux (WSL) which by default does not have access to the drivers for video capture hardware (camera), the display will NOT LOAD while the app is running.

### Recommendation: Run on Native Machine

### Else, there's a complex workaround:

    You can try enabling USB device passthrough to WSL2 using usbipd-win to access the webcam.

    You can pass the webcam device to the Docker container using the --device flag.

    For rendering windows (like OpenCV video frames), you'll need to set up X11 forwarding to display the frames on your Windows machine.

### for reference: https://askubuntu.com/questions/1405903/capturing-webcam-video-with-opencv-in-wsl2


<br>


<hr>