using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;
using Cudafy.Translator;

namespace KnapsackDemo
{
    internal class GpuChunkedBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            long permutations = 2L << count;
            const int gpuBlocks = 1024;
            const int gpuBlockThreads = 1024;
            const int gpuThreads = gpuBlocks*gpuBlockThreads;

            Console.WriteLine("Using {0:N0} gpu threads", gpuThreads);

            int[] weights = new int[count];
            int[] values = new int[count];
            int[] results = new int[gpuThreads];
            for (int i = 0; i < count; i++)
            {
                weights[i] = items[i].Weight;
                values[i] = items[i].Value;
            }

            CudafyTranslator.GenerateDebug = true; // Needed for NSIGHT Cuda debugging
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            // allocate the memory on the GPU
            int[] dev_results = gpu.Allocate<int>(gpuThreads);

            // copy the arrays 'weights' and 'values' to the GPU
            int[] dev_weights = gpu.CopyToDevice(weights);
            int[] dev_values = gpu.CopyToDevice(values);

            long bestPermutation = 0;
            int bestValue = 0;

            for (long n = 0; n < permutations; n += gpuThreads)
            {
                gpu.Launch(gpuBlocks, gpuBlockThreads).add(n, scenario.MaxWeight, dev_weights, dev_values, dev_results);

                // copy the array 'results' back from the GPU to the CPU
                gpu.CopyFromDevice(dev_results, results);

                for (int i = 0; i < gpuThreads; i++)
                {
                    if (results[i] > bestValue)
                    {
                        bestValue = results[i];
                        bestPermutation = i + n;
                    }
                }
            }

            gpu.FreeAll();

            var bestList = PermutationHelper.GetList(items, bestPermutation);
            int bestWeight = bestList.Sum(x => x.Weight);

            var endTime = DateTime.Now;
            return new KnapsackSolution(bestList.ToList(), bestWeight, bestValue, endTime.Subtract(startTime));
        }

        [Cudafy]
        public static void add(GThread thread, long offset, int maxWeight, int[] a, int[] b, int[] c)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x*thread.blockDim.x;
            long permutation = tid + offset;
            
            int items = a.Length;
            int totalValue = 0;
            int totalWeight = 0;
            for (int index = 0; index < items; index++)
            {
                var valueAtBit = permutation & (1L << index);
                if (valueAtBit > 0)
                {
                    totalWeight += a[index];
                    totalValue += b[index];
                }
            }


            if (totalWeight <= maxWeight)
            {
                c[tid] = totalValue;
            }
            else
            {
                c[tid] = -1;
            }
        }
    }
}
