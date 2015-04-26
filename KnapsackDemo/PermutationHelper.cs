using System.Collections.Generic;

namespace KnapsackDemo
{
    internal class PermutationHelper
    {
        public static IList<PackableItem> GetList(PackableItem[] items, long permutation)
        {
            var list = new List<PackableItem>();
            var count = items.Length;
            for (int i = 0; i < count; i++)
            {
                if ((permutation & (1 << i)) > 0)
                {
                    list.Add(items[i]);
                }
            }

            return list;
        }
    }
}