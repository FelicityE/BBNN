GCC = g++ 
FLAGS = -fopenmp -O3
DFLAGS = 
LIBDIRS = -I.


build/main: build/main.o build/ann.o build/utility.o build/activationFunctions.o build/errorFunctions.o
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -o build/main build/main.o build/ann.o build/utility.o build/activationFunctions.o build/errorFunctions.o

build/main.o: source/main.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/main.o source/main.cpp -lm

build/ann.o: source/ann.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/ann.o source/ann.cpp -lm

build/utility.o: source/utility.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/utility.o source/utility.cpp -lm

build/activationFunctions.o: source/activationFunctions.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/activationFunctions.o source/activationFunctions.cpp -lm

build/errorFunctions.o: source/errorFunctions.cpp
	$(GCC) $(FLAGS) $(DFLAGS) $(LIBDIRS) -c -o build/errorFunctions.o source/errorFunctions.cpp -lm

clean:
	rm build/main build/*.o 

# main inlcudes ann.h includes utility.h includes structures.h includes pramas.h