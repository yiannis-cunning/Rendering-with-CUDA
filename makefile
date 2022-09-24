defualt: render
	render

render: fastGraphicsV2.cu linalg.cpp gpuController.cu
	nvcc $^ -lSDL2 -LC:\clibs\SDL2\lib\x64 -IC:\clibs\SDL2\include -o $@
