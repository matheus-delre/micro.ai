namespace Core.Models
{
    public sealed class Result
    {
        public required string Answer { get; init; }
        public required List<ScoredDocument> Context { get; init; }
    }
}
