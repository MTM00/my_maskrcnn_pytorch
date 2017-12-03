all: src/cuda/roi_pooling.cu.o
	python build.py

src/cuda/roi_pooling.cu.o: src/cuda/roi_pooling_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
