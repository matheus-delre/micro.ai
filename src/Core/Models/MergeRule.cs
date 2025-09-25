namespace Core.Models
{
    public sealed class MergeRule
    {
        public required string Left { get; init; }
        public required string Right { get; init; }
        public required string Merge { get; init; }
        public required int Rank { get; init; }
    }
}
