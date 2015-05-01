using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;
using Cudafy.Translator;

namespace KnapsackDemo
{
    internal class GpuNaiveBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            var permutations = 2 << count;

            int[] weights = new int[count];
            int[] values = new int[count];
            int[] results = new int[permutations];
            for (int i = 0; i < count; i++)
            {
                weights[i] = items[i].Weight;
                values[i] = items[i].Value;
            }

            //CudafyTranslator.GenerateDebug = true; // Needed for NSIGHT Cuda debugging but causes slowdown
            CudafyModule km = CudafyTranslator.Cudafy();

            var codeGenerationTime = DateTime.Now.Subtract(startTime);
            Console.WriteLine("Code generation took {0:N1}s", codeGenerationTime.TotalSeconds);
            startTime = DateTime.Now;

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            // allocate the memory on the GPU
            int[] dev_results = gpu.Allocate<int>(permutations);

            // copy the arrays 'a' and 'b' to the GPU
            int[] dev_weights = gpu.CopyToDevice(weights);
            int[] dev_values = gpu.CopyToDevice(values);

            gpu.Launch(128, 128).add(scenario.MaxWeight, dev_weights, dev_values, dev_results);

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_results, results);

            gpu.FreeAll();

            long bestPermutation = 0;
            int bestValue = 0;

            for (int i = 0; i < permutations; i++)
            {
                if (results[i] > bestValue)
                {
                    bestValue = results[i];
                    bestPermutation = i;
                }
            }
            var bestList = PermutationHelper.GetList(items, bestPermutation);
            int bestWeight = bestList.Sum(x => x.Weight);

            var endTime = DateTime.Now;
            return new KnapsackSolution(bestList.ToList(), bestWeight, bestValue, endTime.Subtract(startTime));
        }

        [Cudafy]
        public static void add(GThread thread, int maxWeight, int[] a, int[] b, int[] c)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            if (tid < c.Length)
            {
                int items = a.Length;
                int totalValue = 0;
                int totalWeight = 0;
                for (int index = 0; index < items; index++)
                {
                    var valueAtBit = tid & (1 << index);
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
}
