using Application;
using Application.Embedding;
using Application.LLM;
using Core.Abstractions;
using Core.Options;
using Infrastructure.Persistence;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Infrastructure.Extensions
{
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddMicroAi(this IServiceCollection services, IConfiguration cfg)
        {
            services.Configure<TokenizerOptions>(cfg.GetSection("Options:Tokenizer"));
            services.Configure<PipelineOptions>(cfg.GetSection("Options:Pipeline"));
            services.Configure<StorageOptions>(cfg.GetSection("Options:Storage"));
            services.Configure<RocksDbOptions>(cfg.GetSection("Options:RocksDb"));
            services.Configure<TorchOptions>(cfg.GetSection("Options:Torch"));
            services.Configure<LlmOptions>(cfg.GetSection("Options:Llm"));

            services.AddSingleton<DbFactory>();
            services.AddSingleton<IndexStore>();

            services.AddSingleton<IStateStore>(sp =>
            {
                var storage = sp.GetRequiredService<IOptionsMonitor<StorageOptions>>().CurrentValue;

                return new StateStore(
                        sp.GetRequiredService<ILogger<StateStore>>(),
                        sp.GetRequiredService<IOptionsMonitor<StorageOptions>>(),
                        sp.GetRequiredService<DbFactory>());
            });

            services.AddSingleton<ILanguageModel, TinyGpt>();
            services.AddSingleton<ICheckpointStore, CheckpointStore>();
            services.AddSingleton<IDiagnostics, Diagnostics>();
            services.AddSingleton<IEmbedder, Embedder>();
            services.AddSingleton<IVectorStore, VectorStore>();
            services.AddSingleton<ITokenizer, Tokenizer>();
            services.AddSingleton<IRetriever, Retriever>();
            services.AddSingleton<IReranker, Reranker>();
            services.AddSingleton<IPipeline, Pipeline>();

            return services;
        }
    }
}
