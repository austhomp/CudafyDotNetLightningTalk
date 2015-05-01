﻿using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;
using Cudafy.Translator;

namespace KnapsackDemo
{
    internal class CpuOpenCLBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            long permutations = 2L << count;
            const int blocks = 1024;
            const int blockThreads = 1024;
            const int threads = blocks * blockThreads;

            Console.WriteLine("Using {0:N0} threads (OpenCL threads, not system threads)", threads);

            int[] weights = new int[count];
            int[] values = new int[count];
            int[] results = new int[threads];
            for (int i = 0; i < count; i++)
            {
                weights[i] = items[i].Weight;
                values[i] = items[i].Value;
            }

            CudafyModes.Target = eGPUType.OpenCL;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

            CudafyModule km = CudafyTranslator.Cudafy();

            var codeGenerationTime = DateTime.Now.Subtract(startTime);
            Console.WriteLine("Code generation took {0:N1}s", codeGenerationTime.TotalSeconds);
            startTime = DateTime.Now;

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            // allocate the memory on the GPU
            int[] dev_results = gpu.Allocate<int>(threads);

            // copy the arrays 'weights' and 'values' to the GPU
            int[] dev_weights = gpu.CopyToDevice(weights);
            int[] dev_values = gpu.CopyToDevice(values);

            long bestPermutation = 0;
            int bestValue = 0;

            for (long n = 0; n < permutations; n += threads)
            {
                gpu.Launch(blocks, blockThreads).add(n, scenario.MaxWeight, dev_weights, dev_values, dev_results);

                // copy the array 'results' back from the GPU to the CPU
                gpu.CopyFromDevice(dev_results, results);

                for (int i = 0; i < threads; i++)
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
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
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
