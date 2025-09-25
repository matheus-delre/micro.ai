using Core.Abstractions;
using Core.Models;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Infrastructure.Persistence
{
    public sealed class Retriever(
        ILogger<Retriever> log, 
        ITokenizer bpe,
        IndexStore index, 
        IVectorStore vec, 
        IEmbedder emb, 
        IOptionsMonitor<PipelineOptions> opts) : IRetriever
    {
        private readonly IVectorStore _vec = vec;

        public void Index(IEnumerable<Document> documents)
        {
            foreach (var d in documents)
            {
                index.UpsertDocument(d);
                var ids = bpe.Encode(d.Text);

                index.UpdatePostings(ids, d.Id);
                var vec = emb.Embed(ids);

                _vec.Upsert(d.Id, vec);
            }
            
            log.LogInformation("[Hybrid] Indexed {Count} docs. vectors={VecCount}", documents.Count(), _vec.Count());
        }

        public IEnumerable<ScoredDocument> Retrieve(string query, int topK = 5)
        {
            var ids = bpe.Encode(query);
            var scoresLex = new Dictionary<string, double>();

            var unique = ids.Distinct();

            foreach (var t in unique)
            {
                var bucket = index.GetPosting(t);
                var df = bucket.Count;

                if (df == 0) 
                    continue;

                var idf = Math.Log(1 + (double)Math.Max(1, _vec.Count()) / (1 + df));
                
                foreach (var (doc, tf) in bucket)
                    scoresLex[doc] = scoresLex.GetValueOrDefault(doc) + tf * idf;
            }

            var qvec = emb.Embed(ids);
            var topVec = _vec.TopK(qvec, Math.Max(topK * 4, 10));
            var scoresVec = topVec.ToDictionary(x => x.id, x => (double)x.score);

            double a = opts.CurrentValue is { } o ? o is PipelineOptions ro ? ro.DefaultTopK : 5 : 5; 

            double lexW = (opts.CurrentValue as dynamic)?.LexWeight ?? 0.4;
            double vecW = (opts.CurrentValue as dynamic)?.VecWeight ?? 0.6;

            var idsAll = scoresLex.Keys.Union(scoresVec.Keys);

            var combined = idsAll
                .Select(id => new { id, s = lexW * scoresLex.GetValueOrDefault(id) + vecW * scoresVec.GetValueOrDefault(id) })
                .OrderByDescending(x => x.s)
                .Take(topK)
                .Select(x => new ScoredDocument { Document = index.GetDocument(x.id)!, Score = x.s })
                .Where(sd => sd.Document != null)
                .ToList();

            return combined;
        }
    }
}
