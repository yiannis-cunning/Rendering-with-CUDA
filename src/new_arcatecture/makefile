

run:
	nvcc looper.cpp engineapi.cpp common.cpp controlModes.cpp linalg.cpp sdl_window.cpp renderer.cpp ../gpuController.cu ../File_Conversion/readSTL.cpp -o run -l C:\clibs\SDL2\lib\x64\SDL2



defualt: engineapi.c looper.c common.c
	nvcc $^ -l C:\clibs\SDL2\lib\x64\SDL2 -o run.exe

