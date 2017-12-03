all: roi_pooling_kernel.cu roi_pooling_kernel.h
	nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

clean:
	rm roi_pooling.cu.o
