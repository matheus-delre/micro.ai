namespace Core.Options
{
    public sealed class LlmOptions
    {
        public int VocabSize { get; set; } = 0;
        public int Dim { get; set; } = 256;
        public int Heads { get; set; } = 8; 
        public int Layers { get; set; } = 4;  
        public int MaxSeq { get; set; } = 128;
        public double Dropout { get; set; } = 0.0;
        public int Seed { get; set; } = 42;
        public int Epochs { get; set; } = 200;
        public int BatchSize { get; set; } = 8;
        public double LearningRate { get; set; } = 3e-4;
        public double WeightDecay { get; set; } = 0.0;
        public double GradClip { get; set; } = 1.0;
        public int CurriculumStartSeq { get; set; } = 64;
        public int WarmupSteps { get; set; } = 200;
        public double MinLearningRate { get; set; } = 1e-5;
        public float TopP { get; set; } = 0.0f;
        public int DefaultMaxNewTokens { get; set; } = 64;
        public float Temperature { get; set; } = 0.9f;
        public int TopK { get; set; } = 0;
        public string Device { get; set; } = "cpu";
    }
}
