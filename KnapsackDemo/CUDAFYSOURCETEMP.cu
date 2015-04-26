#include <stdio.h>

// KnapsackDemo.GpuBruteForceSolver
extern "C" __global__  void add( int* scenario, int scenarioLen0,  int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0);

// KnapsackDemo.GpuBruteForceSolver
extern "C" __global__  void add( int* scenario, int scenarioLen0,  int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0)
{
	int num = threadIdx.x + blockIdx.x * blockDim.x;
	printf("%d\n",num);
	c[(num)] = num;
}
