using Core.Abstractions;
using Core.Models;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Application
{
    public sealed class Pipeline(
        ILogger<Pipeline> log, 
        IRetriever retriever, 
        IReranker reranker,
        ILanguageModel lm, 
        ITokenizer tok,
        IOptionsMonitor<PipelineOptions> opts) : IPipeline
    {
        public Result Ask(string query, int topK = 5)
        {
            var k = topK > 0 ? topK : opts.CurrentValue.DefaultTopK;

            var cands = retriever.Retrieve(query, k * 3);
            var ctx = reranker.Rerank(query, cands, k).ToList();

            log.LogInformation("RAG query='{Query}' => contexts={Count}", query, ctx.Count);

            if (ctx.Count == 0)
                return new Result { Answer = "Não encontrei evidências no repositório para responder.", Context = ctx };

            var prompt = "CONTEXTOS:\n" +
                         string.Join("\n", ctx.Select((c, i) => $"[{i + 1}] {c.Document.Text}")) +
                         "\n\nTAREFA: Responda à pergunta usando apenas os CONTEXTOS acima. Se a resposta não estiver neles, diga que não encontrou.\n" +
                         $"PERGUNTA: {query}\nRESPOSTA:";

            var promptIds = tok.Encode(prompt, addBosEos: true);
            var outIds = lm.Generate(promptIds, maxNewTokens: opts.CurrentValue.MaxAnswerTokens, temperature: 0.9f, topK: 0);
            var text = tok.Decode(outIds);

            var suffix = text[Math.Min(tok.Decode(promptIds).Length, text.Length)..];

            return new Result { Answer = string.IsNullOrWhiteSpace(suffix) ? text : suffix.Trim(), Context = ctx };
        }

        public void Index(IEnumerable<Document> documents)
            => retriever.Index(documents);
    }
}
