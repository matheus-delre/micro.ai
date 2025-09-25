namespace Core.Models
{
    public sealed class CheckpointMeta
    {
        public int Version { get; init; } = 1;
        public string Schema { get; init; } = "tensors";
        public IReadOnlyList<string>? Names { get; init; }
        public IReadOnlyList<long[]>? Shapes { get; init; }
    }
}
