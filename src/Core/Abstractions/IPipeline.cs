using Core.Models;
using Document = Core.Models.Document;

namespace Core.Abstractions
{
    public interface IPipeline
    {
        Result Ask(string query, int topK = 5);
        void Index(IEnumerable<Document> documents);
    }
}
