





# gpuController.cu
CFILES = looper.cpp renderer_cuda.cu engineapi.cpp common.cpp controlModes.cpp linalg.cpp sdl_window.cpp renderer.cpp File_Conversion/readSTL.cpp

run: ${CFILES}
	nvcc $^ -o run -l C:\clibs\SDL2\lib\x64\SDL2


game: ${CFILES} ../game/game.cpp
	nvcc $^ -o run -l C:\clibs\SDL2\lib\x64\SDL2