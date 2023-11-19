defualt: run


run: trim.cu linalg.obj gpuController.obj Control\controlModes.cpp File_Conversion\readSTL.cpp 
	nvcc trim.cu ../tmp/linalg.obj ../tmp/gpuController.obj Control\controlModes.cpp File_Conversion\readSTL.cpp  -l C:\clibs\SDL2\lib\x64\SDL2 -o ../bin/$@

linalg.obj: linalg.cpp
	nvcc -c $^ -o ../tmp/linalg.obj

gpuController.obj: gpuController.cu
	nvcc -c $^ -o ../tmp/gpuController.obj

# -lSDL2 -LC:\clibs\SDL2\lib\x64 -IC:\clibs\SDL2\include