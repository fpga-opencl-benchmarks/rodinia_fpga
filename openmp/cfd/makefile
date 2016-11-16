OMP_NUM_THREADS = 8
MIC_OMP_NUM_THREADS = 216

CXXFLAGS= -O3 -lrt -fopenmp

all: euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double euler3d_cpu_offload

euler3d_cpu: euler3d_cpu.cpp
	$(CXX) $(CXXFLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu

euler3d_cpu_offload:
	$(CXX) $(CXXFLAGS) -g -DOMP_OFFLOAD -Dblock_length=$(MIC_OMP_NUM_THREADS) -offload-option,mic,compiler,"-no-opt-prefetch" euler3d_cpu.cpp -o euler3d_cpu_offload

euler3d_cpu_double: euler3d_cpu_double.cpp
	$(CXX) $(CXXFLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	$(CXX) $(CXXFLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	$(CXX) $(CXXFLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double


clean:
	rm -f euler3d_cpu
	rm -f euler3d_cpu_offload
	rm -f euler3d_cpu_double

	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_cpu_double
