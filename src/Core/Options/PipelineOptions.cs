namespace Core.Options
{
    public sealed class PipelineOptions
    {
        public int DefaultTopK { get; set; } = 5;
        public int MaxAnswerTokens { get; set; } = 512;
        public double LexWeight { get; set; } = 0.4;
        public double VecWeight { get; set; } = 0.6;
    }
}
