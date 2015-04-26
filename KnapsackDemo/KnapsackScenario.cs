using System.Collections.Generic;
using System.Linq;

namespace KnapsackDemo
{
    internal class KnapsackScenario
    {
        private readonly IEnumerable<PackableItem> _availableItems;
        private readonly int _maxWeight;

        public KnapsackScenario(IEnumerable<PackableItem> availableItems, int maxWeight)
        {
            _availableItems = availableItems;
            _maxWeight = maxWeight;
        }

        public int MaxWeight
        {
            get { return _maxWeight; }
        }

        public IEnumerable<PackableItem> AvailableItems
        {
            get { return _availableItems.ToList(); }
        }
    }
}