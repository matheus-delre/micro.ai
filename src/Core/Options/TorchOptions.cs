namespace Core.Options
{
    public sealed class TorchOptions
    {
        public string Device { get; set; } = "cpu";
        public int Seed { get; set; } = 42;
        public int Dim { get; set; } = 256;
        public float LearningRate { get; set; } = 1e-3f;
        public int Epochs { get; set; } = 2;
        public int BatchSize { get; set; } = 16;
        public int NegativesPerPositive { get; set; } = 2;
        
    }
}
