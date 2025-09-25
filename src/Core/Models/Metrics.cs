namespace Core.Models
{
    public sealed class Metrics
    {
        public int VocabSize { get; init; }
        public int MergeCount { get; init; }
        public int EncodeCacheSize { get; init; }
        public int DecodeCacheSize { get; init; }
    }
}
