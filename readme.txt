3D rendering project

Using cuda to speed up rendering process.
* still slow as rasterization and filling is done with code -> everything is very inefficient for GPU to do.






FEATURES:
       - Load stl images (No color)
       - Run a window using SDL
       - Control position with WASD and mouse
              - Edit Control/ControlModes.cpp for ControlModes

DEPENDANCIES:
       - SDL2 required
       - MUST LINK SDL libs and include directories in the make file - Only for compling
       - MUST have test.bmp, blaarkop.stl, and SDL2.dll in same folder for 'run.exe' to run - Just for running
       - May get errors for out of date GPU drivers


              
