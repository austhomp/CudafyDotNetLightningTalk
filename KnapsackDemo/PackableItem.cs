namespace KnapsackDemo
{
    internal class PackableItem
    {
        private readonly string _name;
        private readonly int _weight;
        private readonly int _value;

        public PackableItem(string name, int weight, int value)
        {
            _name = name;
            _weight = weight;
            _value = value;
        }

        public string Name
        {
            get { return _name; }
        }

        public int Weight
        {
            get { return _weight; }
        }

        public int Value
        {
            get { return _value; }
        }
    }
}