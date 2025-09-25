using Core.Models;

namespace Core.Abstractions
{
    public interface IReranker
    {
        IEnumerable<ScoredDocument> Rerank(string query, IEnumerable<ScoredDocument> candidates, int topK = 5);
    }
}
