all:
	gcc -Wall -O3 cla-serial.c -o cla-serial -g
	nvcc -O3 cla-parallel.cu -o cla-parallel
