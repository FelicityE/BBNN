GCC = g++ 
FLAGS = -fopenmp -O3
DFLAGS = 
LIBDIRS = -I.


build/main: build/main.o build/ann.o build/utility.o
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -o build/main build/main.o build/ann.o build/utility.o

build/main.o: source/main.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/main.o source/main.cpp -lm

build/ann.o: source/ann.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/ann.o source/ann.cpp -lm

build/utility.o: source/utility.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/utility.o source/utility.cpp -lm

clean:
	rm build/main build/*.o 