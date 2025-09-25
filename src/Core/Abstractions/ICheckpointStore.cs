using Core.Models;

namespace Core.Abstractions
{
    public interface ICheckpointStore
    {
        void SaveNamed(string prefix, IReadOnlyList<CheckpointTensor> tensors, object? optionsSnapshot = null);
        (IReadOnlyList<CheckpointTensor> Tensors, CheckpointMeta Meta)? LoadNamed(string prefix);
        void SaveMatrix(string prefix, float[] weights, int rows, int dim, object? optionsSnapshot = null);
        (float[] Weights, int Rows, int Dim)? LoadMatrix(string prefix);
        void SaveOptions<T>(string prefix, T options);
        T? LoadOptions<T>(string prefix);
    }
}
