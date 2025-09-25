namespace Core.Abstractions
{
    public interface IVectorStore
    {
        void Upsert(string id, ReadOnlySpan<float> vector);
        (string id, float score)[] TopK(ReadOnlySpan<float> query, int k);
        bool TryGet(string id, out float[] vector);
        long Count();
    }
}
