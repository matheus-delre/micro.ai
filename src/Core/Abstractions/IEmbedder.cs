namespace Core.Abstractions
{
    public interface IEmbedder
    {
        float[] Embed(int[] tokenIds);
        (int epochs, double finalLoss) Train((int[] A, int[] B, int Label)[] pairs);
        void Save();
        int Dim { get; }
    }
}
