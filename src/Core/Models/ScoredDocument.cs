namespace Core.Models
{
    public sealed class ScoredDocument
    {
        public required Document Document { get; init; }
        public required double Score { get; init; }
    }
}
