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


            List<PackableItem> items = new List<PackableItem>
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
                new PackableItem("toy triceratops", 10, 19)
            };

            var scenario = new KnapsackScenario(items.Take(14) , 100);
            var cpuResult = cpuBruteForceSolver.Solve(scenario);
            foreach (var item in cpuResult.PackedItems)
            {
                Console.WriteLine("Packed: weight {0}\tvalue {1} \t\"{2}\"", item.Weight, item.Value, item.Name);
            }
            Console.WriteLine();
            Console.WriteLine("Packed {0} items out of {1}", cpuResult.PackedItems.Count(), scenario.AvailableItems.Count());
            Console.WriteLine("Value: {0}\tWeight: {1}\tElapsed {2}", cpuResult.TotalValue, cpuResult.Weight, cpuResult.ElapsedTime);

            IKnapsackSolver gpuBruteForceSolver = new GpuBruteForceSolver();
            var gpuResult = gpuBruteForceSolver.Solve(scenario);

            if (cpuResult.TotalValue == gpuResult.TotalValue && cpuResult.Weight == gpuResult.Weight)
            {
                Console.WriteLine("GPU results match CPU results!");
            }
            else
            {
                Console.WriteLine("Oh noes the GPU does not match :(");
            }

            Console.WriteLine("Packed {0} items out of {1}", gpuResult.PackedItems.Count(), scenario.AvailableItems.Count());
            Console.WriteLine("Value: {0}\tWeight: {1}\tElapsed {2}", gpuResult.TotalValue, gpuResult.Weight, gpuResult.ElapsedTime);

            Console.ReadKey();
        }
    }
}