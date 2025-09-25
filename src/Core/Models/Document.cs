namespace Core.Models
{
    public sealed class Document
    {
        public required string Id { get; init; }
        public required string Text { get; init; }
        public Dictionary<string, string>? Metadata { get; init; }
    }
}
