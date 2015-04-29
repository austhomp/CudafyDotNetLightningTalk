using System;
using System.Collections.Generic;
using System.Linq;

namespace KnapsackDemo
{
    internal class Demo
    {
        public void Execute()
        {
            IKnapsackSolver cpuBruteForceSolver = new CpuBruteForceSolver();

            List<PackableItem> items = GenerateItems();

            var scenario = new KnapsackScenario(items , 100);

            Console.WriteLine("Starting Knapsack solver for {0} items", scenario.AvailableItems.Count());
            Console.WriteLine();

            var cpuResult = cpuBruteForceSolver.Solve(scenario);
            DisplayPackedItems(cpuResult);
            Console.WriteLine();
            Console.WriteLine("Packed {0} items out of {1}", cpuResult.PackedItems.Count(), scenario.AvailableItems.Count());
            Console.WriteLine("Value: {0}\tWeight: {1}\tElapsed {2}", cpuResult.TotalValue, cpuResult.Weight, cpuResult.ElapsedTime);

            IKnapsackSolver gpuBruteForceSolver = new GpuChunkedBruteForceSolver();
            //IKnapsackSolver gpuBruteForceSolver = new GpuSingleShotBruteForceSolver();
            //IKnapsackSolver gpuBruteForceSolver = new GpuNaiveBruteForceSolver();

            Console.WriteLine();
            var gpuResult = gpuBruteForceSolver.Solve(scenario);

            Console.WriteLine("Packed {0} items out of {1}", gpuResult.PackedItems.Count(), scenario.AvailableItems.Count());
            Console.WriteLine("Value: {0}\tWeight: {1}\tElapsed {2}", gpuResult.TotalValue, gpuResult.Weight, gpuResult.ElapsedTime);

            if (cpuResult.TotalValue == gpuResult.TotalValue && cpuResult.Weight == gpuResult.Weight)
            {
                Console.WriteLine("\tGPU results match CPU results!");
            }
            else
            {
                Console.WriteLine("\tOh noes the GPU does not match :(");
            }
            Console.WriteLine();
            var timeDifference = (cpuResult.ElapsedTime - gpuResult.ElapsedTime).TotalSeconds;
            var speedup = cpuResult.ElapsedTime.TotalMilliseconds / gpuResult.ElapsedTime.TotalMilliseconds;
            Console.WriteLine("GPU ran {0:N1}s faster, speedup is {1:N1}x or {2:P} faster", timeDifference, speedup, speedup - 1.0);

            Console.ReadKey();
        }

        private static void DisplayPackedItems(KnapsackSolution result)
        {
            foreach (var item in result.PackedItems)
            {
                Console.WriteLine("Packed: weight {0}\tvalue {1} \t\"{2}\"", item.Weight, item.Value, item.Name);
            }
        }

        private static List<PackableItem> GenerateItems()
        {
            return new List<PackableItem>
            {
                new PackableItem("rune", 1, 20),
                new PackableItem("small bronze shield", 10, 25),
                new PackableItem("candle", 1, 1),
                new PackableItem("torch", 1, 3),
                new PackableItem("wooden staff", 2, 8),
                new PackableItem("short sword", 4, 27),
                new PackableItem("dagger", 1, 7),
                new PackableItem("scroll of kor por", 1, 80),
                new PackableItem("katana", 6, 31),
                new PackableItem("chainmail tunic", 9, 37),
                new PackableItem("black cloak", 2, 18),
                new PackableItem("wooden club", 3, 13),
                new PackableItem("sledge hammer", 12, 29),
                new PackableItem("quarter staff", 7, 23),
                new PackableItem("silver axe", 11, 44),
                new PackableItem("illustrious wand", 3, 52),
                new PackableItem("stove", 50, 80),
                new PackableItem("barrel", 18, 15),
                new PackableItem("wooden throne", 20, 50),
                new PackableItem("marble bust", 21, 31),
                new PackableItem("golden helm", 5, 62),
                new PackableItem("heavy bow of doom", 26, 99),
                new PackableItem("dull cleaver", 2, 3),
                new PackableItem("toy triceratops", 10, 19)
            };
        }
    }
}