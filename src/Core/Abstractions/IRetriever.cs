using Core.Models;
using Document = Core.Models.Document;

namespace Core.Abstractions
{
    public interface IRetriever
    {
        IEnumerable<ScoredDocument> Retrieve(string query, int topK = 5);
        void Index(IEnumerable<Document> documents);
    }
}
