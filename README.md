# AudioPhile
C++ Library for Sound Processing and Editing, mainly meant for pre-processing tasks and fun.
## Introdcution
This is a C++ Library for Audio Processing made to work with PortAudio Sound API. It includes various methods that we can use with an audio sample. This was made for audio pre-processing tasks for machine learning. Although i have not tested all functions, most of them work as intended.

Current Implementation is tested on Linux and WSL.

## How to set-up
Steps to run:
1. run ```make install-deps```.
2. then run ```make test```.
3. Then you will see possible device IDs as outputs, use this index as the ```deviceID``` variable in ```main.cpp```.
4. the run ```make``` to run with main.cpp. If you get any error, try changing ```deviceID```.
5. You can find all documentation in ```main.cpp``` and ```src/Audio.h```.

## Note:
Since, sound buffers are large in size, due to compute constraints, memory management for audio files is hard and cannot be abstracted easily with a wrapper class. I have tried to keep garbage collection as optimized as possible (it's still not the best) but there still remain some issues. So, while working with AudioFiles, try to keep the operations on AudioFiles as close as possible (in code) and try to instantiate new objects in heap.
