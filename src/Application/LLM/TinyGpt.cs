using Core.Abstractions;
using Core.Enums;
using Core.Models;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;
using NN = TorchSharp.torch.nn;
using OPT = TorchSharp.torch.optim;

namespace Application.LLM
{
    public sealed class TinyGpt : ILanguageModel, IDisposable
    {
        private readonly ILogger<TinyGpt> _log;
        private readonly IOptionsMonitor<LlmOptions> _opts;
        private readonly ICheckpointStore _ckpt;
        private readonly ITokenizer _tok;

        private readonly Device _device;

        private readonly int _vocab;
        private readonly int _dim;
        private readonly int _heads;
        private readonly int _layers;
        private readonly int _maxSeq;

        private readonly double _dropout;

        private TorchSharp.Modules.Embedding _tokEmb, _posEmb;
        private readonly List<Block> _blocks = [];

        private LayerNorm _lnF;

        private Tensor _causalMask;

        private sealed class Block : NN.Module
        {
            private readonly int C, H, Hd;

            private readonly LayerNorm ln1;
            private readonly Linear q, k, v, proj;
            private readonly Dropout dropAttn, dropProj;
            private readonly LayerNorm ln2;
            private readonly Linear fc1, fc2;
            private readonly GELU gelu;
            private readonly Dropout dropMlp;

            public Block(string name, int channels, int heads, double dropoutP) : base(name)
            {
                C = channels; 
                H = heads; 
                Hd = C / H; 

                ln1 = NN.LayerNorm([C]);

                q = NN.Linear(C, C, hasBias: true);
                k = NN.Linear(C, C, hasBias: true);
                v = NN.Linear(C, C, hasBias: true);

                proj = NN.Linear(C, C, hasBias: true);

                dropAttn = NN.Dropout(dropoutP);
                dropProj = NN.Dropout(dropoutP);

                ln2 = NN.LayerNorm([C]);
                fc1 = NN.Linear(C, 4 * C, hasBias: true);
                gelu = NN.GELU();
                fc2 = NN.Linear(4 * C, C, hasBias: true);

                dropMlp = NN.Dropout(dropoutP);

                RegisterComponents();
            }

            public Tensor Forward(Tensor x, Tensor causalMask) // x:[B,T,C], mask:[1,1,T,T] bool
            {
                var B = x.shape[0]; var T = x.shape[1]; var C = x.shape[2];

                var xn = ln1.forward(x);

                var Q = q.forward(xn).reshape([B, T, H, Hd]).permute(0, 2, 1, 3);
                var K = k.forward(xn).reshape([B, T, H, Hd]).permute(0, 2, 1, 3);
                var V = v.forward(xn).reshape([B, T, H, Hd]).permute(0, 2, 1, 3);

                var att = matmul(Q, K.transpose(-2, -1)) / Math.Sqrt(Hd);

                var mask = causalMask
                    .slice(2, 0, T, 1) 
                    .slice(3, 0, T, 1);

                var neg = full_like(att, -1e9);
                var maskFloat = mask.to_type(ScalarType.Float32);

                att = att + neg * maskFloat;

                var prob = F.softmax(att, dim: -1);

                prob = dropAttn.forward(prob);

                var y = matmul(prob, V)                            
                         .permute(0, 2, 1, 3)
                         .reshape([B, T, C]); 

                y = proj.forward(y);
                y = dropProj.forward(y);

                var x2 = x + y;
                var zm = ln2.forward(x2);

                zm = fc2.forward(gelu.forward(fc1.forward(zm)));
                zm = dropMlp.forward(zm);

                return x2 + zm;
            }
        }

        public TinyGpt(ILogger<TinyGpt> log,
                       IOptionsMonitor<LlmOptions> opts,
                       ICheckpointStore ckpt,
                       ITokenizer tok)
        {
            _log = log; 
            _opts = opts; 
            _ckpt = ckpt; 
            _tok = tok;

            _device = (opts.CurrentValue.Device?.ToLowerInvariant() == "cuda" && cuda.is_available()) ? CUDA : CPU;

            _vocab = opts.CurrentValue.VocabSize > 0 ? opts.CurrentValue.VocabSize : _tok.GetMetrics().VocabSize;
            _dim = opts.CurrentValue.Dim;
            _heads = opts.CurrentValue.Heads;
            _layers = opts.CurrentValue.Layers;
            _maxSeq = opts.CurrentValue.MaxSeq;
            _dropout = opts.CurrentValue.Dropout;

            BuildModel();

            var snap = _ckpt.LoadNamed("lm");

            if (snap is { } present)
                TryLoadParameters(present.Tensors);
            else
                _log.LogInformation("LLM fresh (no snapshot found).");
        }

        private void BuildModel()
        {
            _tokEmb = NN.Embedding(num_embeddings: _vocab, embedding_dims: _dim);
            _posEmb = NN.Embedding(num_embeddings: _maxSeq, embedding_dims: _dim);

            _blocks.Clear();

            for (int i = 0; i < _layers; i++)
                _blocks.Add(new Block($"blk{i}", _dim, _heads, _dropout));

            _lnF = NN.LayerNorm([_dim]);

            using var tri = triu(ones([_maxSeq, _maxSeq], dtype: ScalarType.Bool), 1);
            
            _causalMask = tri.reshape([1, 1, _maxSeq, _maxSeq]).to(_device);

            _tokEmb.to(_device); 
            _posEmb.to(_device); 
            _lnF.to(_device); 
            
            foreach (var b in _blocks) 
                b.to(_device);

            Eval();

            _log.LogInformation("LLM built: vocab={V}, dim={D}, heads={H}, layers={L}, maxSeq={S}, device={Dev}",
                _vocab, _dim, _heads, _layers, _maxSeq, _device.type);
        }

        private void Eval()
        {
            _tokEmb.eval();
            _posEmb.eval(); 
            _lnF.eval(); 

            foreach (var b in _blocks)
                b.eval();
        }

        private void TrainMode()
        {
            _tokEmb.train(); 
            _posEmb.train(); 
            _lnF.train(); 

            foreach (var b in _blocks) 
                b.train();
        }

        private Tensor Forward(Tensor idx)
        {
            var B = idx.shape[0]; 
            var T = idx.shape[1];

            var pos = arange(0, T, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
            var x = _tokEmb.forward(idx) + _posEmb.forward(pos);

            foreach (var b in _blocks)
                x = b.Forward(x, _causalMask);

            x = _lnF.forward(x);

            using var Wt = _tokEmb.weight.transpose(0, 1);
            
            var logits = matmul(x, Wt); 
            
            return logits;
        }

        public int[] Generate(int[] promptIds, int maxNewTokens, float temperature = 1.0f, int topK = 0, float topP = 0.0f)
        {
            var rng = new Random(_opts.CurrentValue.Seed);

            var ids = promptIds != null ? [.. promptIds] : new List<int>();
            var specials = _tok.GetSpecialTokenMap();

            if (ids.Count == 0 && specials.TryGetValue(nameof(SpecialToken.BOS), out var bos))
                ids.Add(bos);

            var steps = maxNewTokens > 0 ? maxNewTokens : _opts.CurrentValue.DefaultMaxNewTokens;
            
            Eval();

            for (int s = 0; s < steps; s++)
            {
                var start = Math.Max(0, ids.Count - _maxSeq);
                var ctx = ids.Skip(start).Take(_maxSeq).Select(i => (long)i).ToArray();
                
                if (ctx.Length == 0 && specials.TryGetValue(nameof(SpecialToken.BOS), out bos)) 
                    ctx = [bos];

                using var x = tensor(ctx, dtype: ScalarType.Int64, device: _device).reshape([1, ctx.Length]);
                using var logits = Forward(x);
                
                long T = x.shape[1];
                
                using var last = logits.slice(1, T - 1, T, 1).squeeze(1);

                var sp = _tok.GetSpecialTokenMap();
                var ban = BuildGenBanList(sp);

                BanTokensInLogits(last, ban);

                Tensor scaled = (temperature <= 1e-5f) ? last : last / Math.Max(1e-4f, temperature);

                int nextId;

                int V = (int)scaled.shape[1];
                bool useTopSubset = (topK > 0) || (topP > 0f && topP < 1f);

                if (useTopSubset)
                {
                    
                    int k = topK > 0 ? Math.Min(topK, V) : V;

                    nextId = SampleTopKThenTopP(scaled, k, Math.Clamp(topP, 0f, 0.9999f), rng);
                }
                else
                {
                    if (temperature <= 1e-5f)
                    {
                        using var arg = argmax(scaled, dim: 1);
                        nextId = (int)arg.item<long>();
                    }
                    else
                    {
                        using var probs = softmax(scaled, dim: 1);
                        using var sample = multinomial(probs, 1, replacement: false).to(CPU);
                        nextId = (int)sample.item<long>();
                    }
                }

                ids.Add(nextId);

                if (specials.TryGetValue(nameof(SpecialToken.EOS), out var eos) && nextId == eos)
                    break;
            }

            return [.. ids];
        }

        public (int epochs, double finalLoss) TrainSelfSupervised(IEnumerable<int> tokenStream, int? epochs = null)
        {
            var cfg = _opts.CurrentValue;

            int E = epochs ?? cfg.Epochs;
            int B = Math.Max(1, cfg.BatchSize);

            var tokens = tokenStream?.ToArray() ?? [];

            if (tokens.Length < 3) 
                return (0, 0);

            var ds = new PackedDataset(tokens, cfg.Seed);

            int ComputeT(int epoch)
            {
                int start = cfg.CurriculumStartSeq <= 0 ? _maxSeq : Math.Min(cfg.CurriculumStartSeq, _maxSeq);
                int T = (int)Math.Round(start + epoch / Math.Max(1.0, E - 1.0) * (_maxSeq - start));
                
                return Math.Clamp(T, 2, _maxSeq);
            }

            int StepsPerEpochGivenT(int T)
            {
                int N = tokens.Length - 1;
                int starts = (N > T) ? 1 + (N - 1 - T) / T : 0;
                
                return (starts <= 0) ? 0 : (int)Math.Ceiling(starts / (double)B);
            }

            int totalSteps = 0;

            for (int e = 0; e < E; e++) 
                totalSteps += StepsPerEpochGivenT(ComputeT(e));

            totalSteps = Math.Max(1, totalSteps);

            TrainMode();

            using var opt = OPT.Adam(Parameters(), lr: cfg.LearningRate, weight_decay: cfg.WeightDecay);

            double lastLoss = 0;
            double baseLr = cfg.LearningRate;
            double minLr = Math.Min(cfg.MinLearningRate, baseLr);

            long globalStep = 0;

            for (int epoch = 0; epoch < E; epoch++)
            {
                int T = ComputeT(epoch);

                lastLoss = 0;
                int steps = 0;

                
                foreach (var (arrX, arrY) in ds.Batches(B, T, epochs: 1, shuffle: true))
                {
                    int curB = arrX.Length / T;

                    if (curB == 0) 
                        continue;

                    using var X = tensor(arrX.Select(i => (long)i).ToArray(), [curB, T], dtype: ScalarType.Int64, device: _device);
                    using var Yt = tensor(arrY.Select(i => (long)i).ToArray(), [curB, T], dtype: ScalarType.Int64, device: _device);

                    using var logits = Forward(X);
                    using var flatLogits = logits.reshape([curB * T, _vocab]);
                    using var yFlat = Yt.reshape([curB * T]);

                    using var loss = F.cross_entropy(flatLogits, yFlat);

                    opt.zero_grad();
                    loss.backward();

                    if (cfg.GradClip > 0)
                        ClipGradNorm(Parameters(), (float)cfg.GradClip);

                    globalStep++;

                    double lrNow;

                    if (cfg.WarmupSteps > 0 && globalStep <= cfg.WarmupSteps)
                        lrNow = baseLr * (globalStep / (double)Math.Max(1, cfg.WarmupSteps));
                    else
                    {
                        var done = Math.Min(
                            1.0,
                            (globalStep - Math.Max(0, cfg.WarmupSteps)) / Math.Max(1.0, totalSteps - Math.Max(0, cfg.WarmupSteps)));
                        
                        lrNow = minLr + 0.5 * (baseLr - minLr) * (1.0 + Math.Cos(Math.PI * done));
                    }

                    foreach (var pg in opt.ParamGroups) 
                        pg.LearningRate = (float)lrNow;

                    opt.step();

                    lastLoss += loss.ToDouble();
                    steps++;
                }

                var avg = lastLoss / Math.Max(1, steps);
                var curLr = opt.ParamGroups.Any() ? opt.ParamGroups.First().LearningRate : (float)baseLr;  // <- LINQ com ()

                _log.LogInformation("Self epoch {Epoch}/{Total} (T={T}) loss={Loss:F4} lr={LR}",
                    epoch + 1, E, T, avg, curLr);
            }

            Eval();

            return (E, lastLoss);
        }

        public (int epochs, double finalLoss) TrainSFT(
            IEnumerable<(string Prompt, string Answer)> pairs, int? epochs = null)
        {
            var list = pairs?.ToList() ?? [];
            if (list.Count == 0) return (0, 0);

            var cfg = _opts.CurrentValue;

            int E = epochs ?? Math.Max(1, (int)Math.Ceiling(cfg.Epochs * 0.5));
            int B = Math.Max(1, cfg.BatchSize);

            var sp = _tok.GetSpecialTokenMap();

            sp.TryGetValue(nameof(SpecialToken.BOS), out var BOS);
            sp.TryGetValue(nameof(SpecialToken.SEP), out var SEP);
            sp.TryGetValue(nameof(SpecialToken.EOS), out var EOS);

            var samples = new List<(int[] X, int[] Y, int[] Mask)>(); 
            
            foreach (var (pr, ans) in list)
            {
                var p = _tok.Encode("pergunta: " + (pr ?? ""), addBosEos: false);
                var a = _tok.Encode("resposta: " + (ans ?? ""), addBosEos: false);

                var seq = new List<int>();

                if (BOS > 0) 
                    seq.Add(BOS);

                seq.AddRange(p);

                if (SEP > 0) 
                    seq.Add(SEP);

                int answerStart = seq.Count;

                seq.AddRange(a);

                if (EOS > 0) 
                    seq.Add(EOS);

                if (seq.Count < 2) 
                    continue;

                if (seq.Count > _maxSeq) 
                    seq = seq.Take(_maxSeq).ToList();

                var x = seq.Take(seq.Count - 1).ToArray();
                var y = seq.Skip(1).ToArray();

                var mask = new int[x.Length];

                for (int i = 0; i < mask.Length; i++)
                {
                    if (i >= answerStart && i < x.Length) mask[i] = 1;
                }

                samples.Add((x, y, mask));
            }

            if (samples.Count == 0) 
                return (0, 0);

            TrainMode();

            using var opt = OPT.Adam(Parameters(), lr: cfg.LearningRate, weight_decay: cfg.WeightDecay);

            double last = 0;
            long global = 0;

            var rnd = new Random(cfg.Seed);

            for (int e = 0; e < E; e++)
            {
                last = 0;

                samples = [.. samples.OrderBy(_ => rnd.Next())];

                for (int i = 0; i < samples.Count; i += B)
                {
                    var batch = samples.Skip(i).Take(B).ToList();
                    int curB = batch.Count;

                    if (curB == 0) 
                        continue;

                    int T = batch[0].X.Length; 
                                              
                    T = batch.Min(s => s.X.Length);

                    var flatX = new long[curB * T];
                    var flatY = new long[curB * T];
                    var flatM = new float[curB * T];

                    for (int b = 0; b < curB; b++)
                    {
                        var (Xb, Yb, Mb) = batch[b];

                        for (int t = 0; t < T; t++)
                        {
                            flatX[b * T + t] = Xb[t];
                            flatY[b * T + t] = Yb[t];
                            flatM[b * T + t] = Mb[t];
                        }
                    }

                    using var X = tensor(flatX, [curB, T], dtype: ScalarType.Int64, device: _device);
                    using var Y = tensor(flatY, [curB, T], dtype: ScalarType.Int64, device: _device);
                    using var M = tensor(flatM, [curB, T], dtype: ScalarType.Float32, device: _device);

                    using var logits = Forward(X);
                    using var logp = F.log_softmax(logits, -1);
                    using var pick = logp.gather(-1, Y.unsqueeze(-1)).squeeze(-1); 
                    using var nll = -pick;
                    using var masked = nll * M;

                    var denom = M.sum().ToDouble();
                    
                    if (denom < 1e-8) 
                        continue;

                    using var loss = masked.sum() / denom;
                    
                    global++;
                    
                    var totalSteps = (long)Math.Ceiling((double)samples.Count / B) * E;
                    
                    var lrNow = ComputeWarmupCosine(
                        baseLr: cfg.LearningRate,
                        minLr: Math.Min(cfg.MinLearningRate, cfg.LearningRate),
                        step: global,
                        warmupSteps: cfg.WarmupSteps,
                        totalSteps: totalSteps
                    );

                    foreach (var pg in opt.ParamGroups) 
                        pg.LearningRate = (float)lrNow;

                    opt.zero_grad();
                    loss.backward();

                    if (cfg.GradClip > 0) 
                        ClipGradNorm(Parameters(), (float)cfg.GradClip);

                    opt.step();

                    last += loss.ToDouble();
                }

                _log.LogInformation("SFT epoch {E}/{Tot} loss={L:F4}", e + 1, E, last);
            }

            Eval();

            return (E, last);
        }

        public void Save()
        {
            var tensors = NamedParameters() 
                .Select(p =>
                {
                    using var cpu = p.tensor.detach().cpu().to_type(ScalarType.Float32);
                    
                    long len = 1; 
                    
                    foreach (var s in cpu.shape) 
                        len *= s;

                    var data = new float[len];
                    cpu.data<float>().CopyTo(data);

                    return new CheckpointTensor( 
                        Name: p.name,
                        Data: data,
                        Shape: [.. cpu.shape]
                    );
                })
                .ToList();

                var snap = new { vocab = _vocab, dim = _dim, heads = _heads, layers = _layers, maxSeq = _maxSeq };

                _ckpt.SaveNamed("lm", tensors, snap);

                _log.LogInformation("LLM parameters saved.");
        }

        private static double ComputeWarmupCosine(
            double baseLr, double minLr, long step, long warmupSteps, long totalSteps)
        {
            if (warmupSteps > 0 && step <= warmupSteps)
                return baseLr * (step / (double)Math.Max(1, warmupSteps));

            var w = Math.Max(0, warmupSteps);
            var done = Math.Min(1.0, (step - w) / Math.Max(1.0, totalSteps - w));

            return minLr + 0.5 * (baseLr - minLr) * (1.0 + Math.Cos(Math.PI * done));
        }

        private static int SampleTopKThenTopP(Tensor logitsRow, int k, float topP, Random rng)
        {
            var (topVals, topIdx) = topk(logitsRow, k, dim: 1, largest: true, sorted: true);

            try
            {
                using var pk = softmax(topVals, dim: 1).to(CPU);

                int kk = (int)pk.shape[1];

                var probs = new float[kk];

                pk.data<float>().CopyTo(probs);

                var idsT = topIdx.to(CPU);
                var ids = new long[kk];

                idsT.data<long>().CopyTo(ids);

                int m = kk;

                if (topP > 0f && topP < 1f)
                {
                    double cum = 0.0;

                    for (int i = 0; i < kk; i++)
                    {
                        cum += probs[i];

                        if (cum >= topP) 
                        { 
                            m = i + 1;
                            break; 
                        }
                    }

                    m = Math.Max(1, m);
                }

                double sum = 0.0;

                for (int i = 0; i < m; i++) 
                    sum += probs[i];

                if (sum <= 0) 
                    return (int)ids[0];

                double r = rng.NextDouble() * sum;
                double c = 0.0;

                for (int i = 0; i < m; i++)
                {
                    c += probs[i];

                    if (r <= c) 
                        return (int)ids[i];
                }

                return (int)ids[m - 1];
            }
            finally
            {
                topVals.Dispose();
                topIdx.Dispose();
            }
        }

        private static void ClipGradNorm(IEnumerable<Parameter> parameters, float maxNorm, float eps = 1e-6f)
        {
            double total = 0.0;

            var grads = new List<Tensor>();

            foreach (var p in parameters)
            {
                var g = p.grad;

                if (g is null) 
                    continue;

                grads.Add(g);

                using var n = g.norm();

                total += Math.Pow(n.ToDouble(), 2.0);
            }

            var totalNorm = Math.Sqrt(total);

            if (totalNorm <= maxNorm || totalNorm <= eps) 
                return;

            var scale = (float)(maxNorm / (totalNorm + eps));

            foreach (var g in grads)
                g.mul_(scale);
        }

        private static List<int> BuildGenBanList(IReadOnlyDictionary<string, int> sp)
        {
            var ban = new List<int>();

            void add(string k) 
            { 
                if (sp.TryGetValue(k, out var id)) 
                    ban.Add(id); 
            }
            
            add(nameof(SpecialToken.UNK));
            add(nameof(SpecialToken.MASK));
            add(nameof(SpecialToken.GARBLE));
            add(nameof(SpecialToken.SEP));
            add(nameof(SpecialToken.PAD));

            return ban;
        }

        private static void BanTokensInLogits(Tensor logitsRow, IReadOnlyList<int> banIds)
        {
            if (banIds == null || banIds.Count == 0) 
                return;
            var V = (int)logitsRow.shape[1];

            using var mask = torch.zeros([1, V], dtype: ScalarType.Float32, device: logitsRow.device);
            
            foreach (var id in banIds)
            {
                if (id >= 0 && id < V)
                    mask[0, id] = 1.0f;
            }

            logitsRow -= (1e9f) * mask;
        }

        private IEnumerable<Parameter> Parameters()
        {
            foreach (var p in _tokEmb.parameters()) 
                yield return p;

            foreach (var p in _posEmb.parameters())
                yield return p;

            foreach (var b in _blocks)
                foreach (var p in b.parameters()) 
                    yield return p;

            foreach (var p in _lnF.parameters()) 
                yield return p;
        }

        private List<(string name, Tensor tensor)> NamedParameters()
        {
            var list = new List<(string, Tensor)>();

            int i = 0;

            foreach (var p in Parameters())
                list.Add(($"p{i++:D4}", p));

            return list;
        }

        private void TryLoadParameters(IReadOnlyList<CheckpointTensor> tensors)
        {
            var current = Parameters().ToList(); 

            if (current.Count != tensors.Count)
            {
                _log.LogWarning("LLM snapshot incompatível: on-disk={A}, in-model={B}. Ignorando load.",
                    tensors.Count, current.Count);
                return;
            }

            using var _ = no_grad();

            for (int i = 0; i < current.Count; i++)
            {
                var src = tensors[i];                 
                var dst = current[i];

                using var t = tensor(src.Data, src.Shape, ScalarType.Float32, _device);

                dst.detach().copy_(t);
            }

            _log.LogInformation("LLM parameters loaded.");
        }

        public void Dispose()
        {
            _causalMask?.Dispose();
            _tokEmb?.Dispose(); 
            _posEmb?.Dispose(); 
            _lnF?.Dispose(); 

            foreach (var b in _blocks) 
                b?.Dispose();
        }
    }
}
