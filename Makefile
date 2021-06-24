CXXFLAGS=-std=c++11 -O3 -march=native # -DDEBUG_MODE
HEADERS=$(wildcard include/*.h)
# object files the benchmark depends on
OBJ=src/setup.o src/utils.o
export CXX=mpic++ # mpiicc if using intel

all: benchmark

src/%.o: src/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

%: src/%.o $(HEADERS) $(OBJ)
	$(CXX) $(CXXFLAGS) -Iinclude $< $(OBJ) -o $@

clean:
	\rm -f src/*.o
	\rm -f benchmark
