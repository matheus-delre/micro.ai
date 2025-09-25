namespace Core.Models
{
    public sealed record CheckpointTensor(
        string Name,
        float[] Data,
        long[] Shape
    );
}
