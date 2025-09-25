namespace Core.Models
{
    public sealed class State
    {
        public Dictionary<string, int> Vocabulary { get; init; } = [];
        public List<MergeRule> Merges { get; init; } = [];
        public Dictionary<string, int> SpecialTokenMap { get; init; } = [];
        public int Version { get; init; } = 1;
    }
}
