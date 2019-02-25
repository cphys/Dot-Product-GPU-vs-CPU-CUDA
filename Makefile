#Makefile
#define variables
objects= main.o kernels.o 
NVCC= nvcc               #cuda c compiler
execname= main2

#compile
$(execname): $(objects)
	$(NVCC) -o $(execname) $(objects) 

kernels.o: kernels.cu
	$(NVCC) -c kernels.cu
main.o: main.cu
	$(NVCC) -c main.cu


#clean Makefile
clean:
	rm $(objects)
#end of Makefile

