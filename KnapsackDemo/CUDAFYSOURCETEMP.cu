
// KnapsackDemo.GpuBruteForceSolver
extern "C" __global__  void add(int chunkSize, int maxWeight,  int* weights, int weightsLen0,  int* values, int valuesLen0,  int* results, int resultsLen0, int resultsLen1);

// KnapsackDemo.GpuBruteForceSolver
extern "C" __global__  void add(int chunkSize, int maxWeight,  int* weights, int weightsLen0,  int* values, int valuesLen0,  int* results, int resultsLen0, int resultsLen1)
{
	int num = threadIdx.x + blockIdx.x * blockDim.x;
	results[(num) * resultsLen1 + (0)] = 0;
	results[(num) * resultsLen1 + (1)] = 0;
	int num2 = valuesLen0;
	int num3 = 0;
	int num4 = 0;
	int i = num * chunkSize;
	int num5 = i + chunkSize;
	while (i < num5)
	{
		int num6 = 0;
		int num7 = 0;
		for (int j = 0; j < num2; j++)
		{
			int num8 = i & 1 << (j & 31);
			if (num8 > 0)
			{
				num6 += values[(j)];
				num7 += weights[(j)];
			}
		}
		if (num7 <= maxWeight && num6 > num4)
		{
			num3 = i;
			num4 = num6;
		}
		i++;
	}
	results[(num) * resultsLen1 + (0)] = num3;
	results[(num) * resultsLen1 + (1)] = num4;
}
