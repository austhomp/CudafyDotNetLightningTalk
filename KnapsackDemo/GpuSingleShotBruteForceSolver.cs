using System;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace KnapsackDemo
{
    internal class GpuSingleShotBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            var permutations = 2 << count;
            const int gpuBlocks = 1024;
            var chunkSize = permutations / gpuBlocks + ((permutations % gpuBlocks > 0) ? 1 : 0);

            int[] weights = new int[count];
            int[] values = new int[count];
            int[,] results = new int[gpuBlocks, 2];
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
            int[,] dev_results = gpu.Allocate<int>(gpuBlocks, 2);
            
            // copy the arrays 'weights' and 'values' to the GPU
            int[] dev_weights = gpu.CopyToDevice(weights);
            int[] dev_values = gpu.CopyToDevice(values);

            gpu.Launch(gpuBlocks, 1).add(chunkSize, scenario.MaxWeight, dev_weights, dev_values, dev_results);

            // copy the array 'results' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_results, results);

            gpu.FreeAll();

            long bestPermutation = 0;
            int bestValue = 0;

            for (int i = 0; i < gpuBlocks; i++)
            {
                if (results[i,1] > bestValue)
                {
                    bestValue = results[i, 1];
                    bestPermutation = results[i,0];
                }
            }
            var bestList = PermutationHelper.GetList(items, bestPermutation);
            int bestWeight = bestList.Sum(x => x.Weight);

            var endTime = DateTime.Now;
            return new KnapsackSolution(bestList.ToList(), bestWeight, bestValue, endTime.Subtract(startTime));
        }

        [Cudafy]
        public static void add(GThread thread, int chunkSize, int maxWeight, int[] weights, int[] values, int[,] results)
        {
            int tid = (thread.threadIdx.x + thread.blockIdx.x*thread.blockDim.x);
            results[tid,0] = 0;
            results[tid,1] = 0;

            int itemCount = values.Length;
            int bestPermutation = 0;
            int bestValue = 0;

            int permutation = tid * chunkSize;
            int lastPermutation = permutation + chunkSize;

            while (permutation < lastPermutation)
            {
                int totalValue = 0;
                int totalWeight = 0;
                for (int index = 0; index < itemCount; index++)
                {
                    var valueAtBit = permutation & (1 << index);

                    if (valueAtBit > 0)
                    {
                        totalValue += values[index];
                        totalWeight += weights[index];
                    }
                }

                if (totalWeight <= maxWeight && totalValue > bestValue)
                {
                    bestPermutation = permutation;
                    bestValue = totalValue;
                }

                permutation++;
            }

            results[tid, 0] = bestPermutation;
            results[tid, 1] = bestValue;
        }
    }
}