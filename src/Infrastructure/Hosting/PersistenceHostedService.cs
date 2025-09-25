using Core.Abstractions;
using Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Infrastructure.Hosting
{
    public sealed class PersistenceHostedService(
        ILogger<PersistenceHostedService> log, 
        ITokenizer bpe, 
        IStateStore store) : IHostedService
    {
        public Task StartAsync(CancellationToken cancellationToken)
        {
            log.LogInformation("BPE persistence service started (load handled by tokenizer constructor)");
            
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            var state = new State
            {
                Vocabulary = bpe.GetVocabulary().ToDictionary(kv => kv.Key, kv => kv.Value),
                Merges = [.. bpe.GetMerges().Select((m, i) => new MergeRule { Left = m.Left, Right = m.Right, Merge = m.Merge, Rank = i })],
                SpecialTokenMap = bpe.GetSpecialTokenMap().ToDictionary(kv => kv.Key, kv => kv.Value),
            };

            store.Save(state);
            
            log.LogInformation("BPE persistence service saved state on shutdown.");
            
            return Task.CompletedTask;
        }
    }
}
