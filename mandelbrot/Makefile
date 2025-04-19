DEFS=-DMAXITER=8000 -DESCAPE_RADIUS_SQUARED=16*16

CC=g++
CXX=/opt/homebrew/bin/g++-14
CPPFLAGS = -Wall -Wextra $(DEFS) -Wno-clobbered 
CXXFLAGS = -std=c++14 -Ofast -march=native
LDLIBS   = -pthread $(shell /opt/homebrew/bin/pkg-config sdl2 --libs) 
CPPFLAGS +=         $(shell /opt/homebrew/bin/pkg-config sdl2 --cflags --libs)


BINARIES = mandelbrot mandelbrot-glsl mandelbrot-opencl

all: $(BINARIES)

mandelbrot-glsl: mandelbrot-glsl.cpp
	$(CXX) -o $@ $^ $(CPPFLAGS) $(CXXFLAGS) $(LDLIBS) $(DEFS) -framework OpenGL -lGLEW -g

mandelbrot: mandelbrot.cpp
	$(CXX) -o $@ $^ $(CPPFLAGS) $(CXXFLAGS) $(LDLIBS) $(DEFS) 

mandelbrot-opencl: mandelbrot-opencl.cpp
	clang++ -o $@ $^ $(CPPFLAGS) $(CXXFLAGS) $(LDLIBS) $(DEFS) -framework OpenCL

clean:
	rm -f $(BINARIES)

