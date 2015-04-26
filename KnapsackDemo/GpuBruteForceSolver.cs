using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;
using Cudafy.Translator;

namespace KnapsackDemo
{
    internal class GpuBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            var permutations = (int)Math.Pow(2, count);

            int[] weights = new int[count];
            int[] values = new int[count];
            int[] results = new int[permutations];
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
            int[] dev_c = gpu.Allocate<int>(permutations);
            
            // copy the arrays 'a' and 'b' to the GPU
            int[] dev_a = gpu.CopyToDevice(weights);
            int[] dev_b = gpu.CopyToDevice(values);

            gpu.Launch(128, 128).add(scenario.MaxWeight, dev_a, dev_b, dev_c);

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_c, results);

            gpu.FreeAll();

            long best = 0;
            int bestValue = 0;

            for (int i = 0; i < permutations; i++)
            {
                if (results[i] > bestValue)
                {
                    bestValue = results[i];
                    best = i;
                }
            }
            var bestList = PermutationHelper.GetList(items, best);
            int bestWeight = bestList.Sum(x => x.Weight);

            var endTime = DateTime.Now;
            return new KnapsackSolution(bestList.ToList(), bestWeight, bestValue, endTime.Subtract(startTime));
        }

        [Cudafy]
        public static void add(GThread thread, int maxWeight, int[] a, int[] b, int[] c)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            //Debug.WriteLine("%d", tid);
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