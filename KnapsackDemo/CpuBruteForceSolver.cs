using System;
using System.Linq;

namespace KnapsackDemo
{
    internal class CpuBruteForceSolver : IKnapsackSolver
    {
        public KnapsackSolution Solve(KnapsackScenario scenario)
        {
            var startTime = DateTime.Now;
            var items = scenario.AvailableItems.ToArray();
            var count = items.Length;
            long permutations = 2L << count;
            long best = 0;
            int bestValue = 0;
            int bestWeight = 0;

            for (long permutation = 0; permutation < permutations; permutation++)
            {
                int totalValue = 0;
                int totalWeight = 0;
                for (int index = 0; index < count; index++)
                {
                    var valueAtBit = permutation & (1L << index);
                    if (valueAtBit > 0)
                    {
                        totalValue += items[index].Value;
                        totalWeight += items[index].Weight;
                    }
                }

                if (totalWeight <= scenario.MaxWeight && totalValue > bestValue)
                {
                    best = permutation;
                    bestValue = totalValue;
                    bestWeight = totalWeight;
                    //Console.WriteLine("\tNew high score\tvalue {0}\tweight {1}", totalValue, totalWeight);
                }
            }
            var endTime = DateTime.Now;
            var bestList = PermutationHelper.GetList(items, best);
            return new KnapsackSolution(bestList.ToList(), bestWeight, bestValue, endTime.Subtract(startTime));
        }
    }
}