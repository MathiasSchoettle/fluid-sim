# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -arch=sm_89 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -O3 --use_fast_math --keep --resource-usage

# Libraries
LDLIBS = -lbigduckgl -limgui -lGLEW -lglfw -lGL -lassimp

# Source files
CPP_SOURCES = src/main.cpp src/simulation.cpp src/quad.cpp src/box.cpp
CUDA_SOURCES = src/simulation-cuda.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Header file dependencies
CPP_DEPENDENCIES = $(CPP_SOURCES:.cpp=.d)
CUDA_DEPENDENCIES = $(CUDA_SOURCES:.cu=.d)

# Targets
all: fluid

fluid: $(CPP_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Include header file dependencies
-include $(CPP_DEPENDENCIES)
-include $(CUDA_DEPENDENCIES)

# Generate header file dependencies
%.d: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< -MF $@

%.d: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -M -MT $(@:.d=.o) $< -MF $@

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CUDA_OBJECTS) fluid $(CPP_DEPENDENCIES) $(CUDA_DEPENDENCIES)
