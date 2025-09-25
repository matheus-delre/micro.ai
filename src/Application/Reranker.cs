using Core.Abstractions;
using Core.Models;

namespace Application
{
    public sealed class Reranker(ITokenizer bpe) : IReranker
    {
        public IEnumerable<ScoredDocument> Rerank(string query, IEnumerable<ScoredDocument> candidates, int topK = 5)
        {
            var q = bpe.Encode(query).ToHashSet();

            return [.. candidates
                .Select(c => new ScoredDocument
                {
                    Document = c.Document,
                    Score = c.Score + bpe.Encode(c.Document.Text).Count(q.Contains)
                })
                .OrderByDescending(s => s.Score)
                .Take(topK)];
        }
    }
}
