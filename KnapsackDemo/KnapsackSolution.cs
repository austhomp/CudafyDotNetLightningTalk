using System;
using System.Collections.Generic;
using System.Linq;

namespace KnapsackDemo
{
    internal class KnapsackSolution
    {
        private readonly IEnumerable<PackableItem> _packedItems;
        private readonly int _weight;
        private readonly int _totalValue;
        private readonly TimeSpan _elapsedTime;

        public KnapsackSolution(IEnumerable<PackableItem> packedItems, int weight, int totalValue, TimeSpan elapsedTime)
        {
            _packedItems = packedItems;
            _weight = weight;
            _totalValue = totalValue;
            _elapsedTime = elapsedTime;
        }

        public IEnumerable<PackableItem> PackedItems
        {
            get { return _packedItems.ToList(); }
        }

        public int Weight
        {
            get { return _weight; }
        }

        public TimeSpan ElapsedTime
        {
            get { return _elapsedTime; }
        }

        public int TotalValue
        {
            get { return _totalValue; }
        }
    }
}