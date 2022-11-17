defualt: run


run: trim.cu linalg.obj gpuController.obj Control\controlModes.cpp File_Conversion\readSTL.cpp
	nvcc $^ -o $@

linalg.obj: linalg.cpp
	nvcc -c $^ -o $@

gpuController.obj: gpuController.cu
	nvcc -c $^ -o $@

# -lSDL2 -LC:\clibs\SDL2\lib\x64 -IC:\clibs\SDL2\include