3D rendering project

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


MANIFEST:
       - gpuController.cu + gpuController.h      # For GPU rendering code
       - linalg.cpp + linalg.h                   # quick homemade library for vector manipulation (float3)
       - makefile                                # for make - used with MINGW32-make in CMD (make sure include path to gcc make thing) - complied with nvcc
       - SDL2.dll                                # SDL2 library - 64 bit version
       - test.bmp                                # not actually a test - nessasary for the window initialization
       - trim.cu                                 # main file/Entry point
       - Control/
              - controlModes.cpp + controlModes.h              # Contains input control and movment rules
       - File_Conversion/
              - readSTL.cpp + readSTL.h                        # Contains code for inputing stls
       - blaarkop.stl                                          # Test stl file of a cow