using Application.LLM;
using Core.Abstractions;
using Core.Enums;
using Core.Models;
using Infrastructure.Extensions;
using Infrastructure.Hosting;
using Infrastructure.Persistence;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();

Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    .CreateLogger();

builder.Host.UseSerilog();

builder.Services.AddMicroAi(builder.Configuration);
builder.Services.AddHostedService<PersistenceHostedService>();

builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { Title = "Micro.Ai", Version = "v1" });
});

builder.Services.AddHealthChecks();

var app = builder.Build();

app.UseSerilogRequestLogging();

app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Micro.Ai v1");
    c.DocumentTitle = "Micro.Ai — API";
});

app.MapHealthChecks("/healthz");

var bpe = app.MapGroup("/bpe");

bpe.MapPost("/vocabulary", (ITokenizer tok, InputTrain req) =>
{
    var added = tok.Train(req.Corpus ?? [], req.NumMerges);

    return Results.Ok(new 
    { 
        added, 
        metrics = tok.GetMetrics() 
    });
});


bpe.MapPost("/vocabulary/save", (ITokenizer tok, IStateStore store) =>
{
    var snap = tok.Snapshot();

    store.Save(snap);

    return Results.Ok(new 
    { 
        saved = true, 
        metrics = tok.GetMetrics() 
    });
});

bpe.MapPost("/vocabulary/load", (ITokenizer tok, IStateStore store) =>
{
    if (!store.TryLoad(out var state))
        return Results.NotFound(new { loaded = false, reason = "no state found in RocksDB" });

    tok.Restore(state);

    return Results.Ok(new 
    { 
        loaded = true, 
        metrics = tok.GetMetrics() 
    });
});

bpe.MapGet("/vocabulary/state", (ITokenizer tok) => Results.Ok(new
{
    vocabSize = tok.GetMetrics().VocabSize,
    mergeCount = tok.GetMetrics().MergeCount,
    specials = tok.GetSpecialTokenMap()
}));

var rag = app.MapGroup("/rag");

rag.MapPost("/index/corpus", (
    ITokenizer tok,
    IEmbedder emb,
    IndexStore idx,
    IVectorStore vec,
    IndexCorpusRequest req) =>
{
    var corpus = req.Corpus ?? [];

    if (corpus.Count == 0)
        return Results.Ok(new { indexed = 0 });

    var prefix = string.IsNullOrWhiteSpace(req.Prefix) ? "doc:" : req.Prefix;

    int n = 0;

    for (int i = 0; i < corpus.Count; i++)
    {
        var id = $"{prefix}{i + 1}";
        var text = corpus[i] ?? string.Empty;

        idx.UpsertDocument(new Document { Id = id, Text = text });

        var ids = tok.Encode(text, false);

        idx.UpdatePostings(ids, id);

        var v = emb.Embed(ids);

        vec.Upsert(id, v);

        n++;
    }

    return Results.Ok(new 
    { 
        indexed = n 
    });
});

rag.MapPost("/ask", (IPipeline pipeline, InputQuery req) =>
{
    var res = pipeline.Ask(req.Query ?? string.Empty, req.TopK);

    return Results.Ok(res);
});

var torch = app.MapGroup("/torch");

torch.MapPost("/train", (
    ITokenizer tok,
    IEmbedder trainer,
    InputEmbTrainRequest req) =>
{

    var pairs = (req.Pairs ?? [])
                .Select(p => (
                    A: tok.Encode(p.A ?? "", false),
                    B: tok.Encode(p.B ?? "", false),
                    Label: p.Label >= 0 ? 1 : -1
                )).ToArray();

    var (epochs, loss) = trainer.Train(pairs);

    trainer.Save();

    return Results.Ok(new 
    { 
        trainedEpochs = epochs, 
        finalLoss = loss 
    });
});

torch.MapPost("/save", (IEmbedder trainer) => 
{ 
    trainer.Save(); 
    
    return Results.Ok(new 
    { 
        saved = true 
    }); 
});

torch.MapPost("/rebuild", (
    ITokenizer tok,
    IEmbedder emb,
    IndexStore idx,
    IVectorStore vec) =>
{
    var rebuilt = 0;

    foreach (var d in idx.EnumerateDocuments())
    {
        var ids = tok.Encode(d.Text ?? string.Empty, false);
        var v = emb.Embed(ids);

        vec.Upsert(d.Id, v);

        rebuilt++;
    }

    return Results.Ok(new 
    { 
        rebuilt, 
        dim = emb.Dim 
    });
});

var llm = app.MapGroup("/llm");

llm.MapPost("/train/self", (ITokenizer tok, ILanguageModel lm, InputLlmTrain req) =>
{
    var stream = BuildSelfStream(req.Corpus ?? [], tok);
    
    var (epochs, loss) = lm.TrainSelfSupervised(stream);

    lm.Save();

    return Results.Ok(new 
    { 
        trainedEpochs = epochs, 
        finalLoss = loss 
    });
});

llm.MapPost("/train/sft", (ITokenizer tok, ILanguageModel lm, InputLlmTrainSft req) =>
{
    var pairs = (req.Pairs ?? []).Select(p => (p.Prompt ?? "", p.Answer ?? "")).ToList();
    
    var (e, l) = (lm as TinyGpt)!.TrainSFT(pairs, req.Epochs);
    
    (lm as TinyGpt).Save();

    return Results.Ok(new 
    { 
        trainedEpochs = e, 
        finalLoss = l 
    });
});

llm.MapPost("/generate", (ITokenizer tok, ILanguageModel lm, InputLlmGenerate req) =>
{
    var ids = tok.Encode(req.Prompt ?? string.Empty, addBosEos: true);

    var outIds = lm.Generate(ids, req.MaxNewTokens, req.Temperature, req.TopK, req.TopP);
    
    var text = tok.Decode(outIds);

    return Results.Ok(new 
    { 
        ids = outIds, 
        text 
    });
});

llm.MapPost("/save", (ILanguageModel lm) => 
{ 
    lm.Save(); 
    
    return Results.Ok(new 
    { 
        saved = true 
    }); 
});

static IEnumerable<int> BuildSelfStream(IEnumerable<string> corpus, ITokenizer tok)
{
    var sp = tok.GetSpecialTokenMap();
    var hasSep = sp.TryGetValue(nameof(SpecialToken.SEP), out var SEP);

    foreach (var s in corpus ?? [])
    {
        var ids = tok.Encode(s ?? string.Empty, addBosEos: true);

        foreach (var id in ids) 
            yield return id;

        if (hasSep) 
            yield return SEP;
    }
}

app.Run();

public record InputTrain(List<string> Corpus, int NumMerges = 1000);
public record InputQuery(string Query, int TopK = 5);
public record InputEmbPair(string A, string B, int Label);
public record InputEmbTrainRequest(InputEmbPair[] Pairs);
public record IndexCorpusRequest(List<string> Corpus, string? Prefix = "doc:");
public record InputLlmTrain(List<string> Corpus);
public record InputLlmGenerate(string Prompt, int MaxNewTokens = 64, float Temperature = 0.9f, int TopK = 0, float TopP = 0.0f);
public record InputSftPair(string Prompt, string Answer);
public record InputLlmTrainSft(List<InputSftPair> Pairs, int? Epochs = null);