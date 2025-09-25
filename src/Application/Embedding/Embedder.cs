using Core.Abstractions;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;
using NN = TorchSharp.torch.nn;
using OPT = TorchSharp.torch.optim;

namespace Application.Embedding
{
    public sealed class Embedder : IEmbedder, IDisposable
    {
        private readonly ILogger<Embedder> _log;
        private readonly IOptionsMonitor<TorchOptions> _opts;
        private readonly ICheckpointStore _store;
        private readonly Device _device;

        private TorchSharp.Modules.Embedding? _emb;
        private int _rows;
        private int _dim;

        public int Dim 
            => _dim;

        public Embedder(
            ILogger<Embedder> log,
            IOptionsMonitor<TorchOptions> opts,
            ICheckpointStore store)
        {
            _log = log; 
            _opts = opts; 
            _store = store;
            _device = opts.CurrentValue.Device?.ToLowerInvariant() == "cuda" && cuda.is_available() ? CUDA : CPU;
            
            InitializeFromStoreOrFresh();
        }

        private void InitializeFromStoreOrFresh()
        {
            var cfg = _opts.CurrentValue;

            _dim = cfg.Dim > 0 ? cfg.Dim : 256;

            var loaded = _store.LoadMatrix("emb");

            if (loaded is { } lw)
            {
                _rows = lw.Rows; 
                _dim = lw.Dim;

                _emb = NN.Embedding(num_embeddings: _rows, embedding_dims: _dim);
                _emb.to(_device);

                using var w = tensor(
                    lw.Weights,
                    [_rows, _dim],
                    dtype: ScalarType.Float32,
                    device: _device
                );

                using var _ = no_grad();

                _emb.weight.detach().copy_(w);
                
                _log.LogInformation("TorchEmbedder loaded from Rocks: rows={Rows}, dim={Dim}, device={Device}", _rows, _dim, _device.type);
            }
            else
            {
                _rows = 1024;
                _emb = NN.Embedding(num_embeddings: _rows, embedding_dims: _dim);
                _emb.to(_device);

                _log.LogInformation("TorchEmbedder initialized fresh: rows={Rows}, dim={Dim}, device={Device}", _rows, _dim, _device.type);
            }
        }

        private static int NextPow2(int v)
        {
            v--; 
            v |= v >> 1; 
            v |= v >> 2; 
            v |= v >> 4; 
            v |= v >> 8; 
            v |= v >> 16; 
            v++;
            
            return v;
        }

        private void EnsureRows(int requiredMaxId)
        {
            if (requiredMaxId < _rows) 
                return;

            var newRows = NextPow2(requiredMaxId + 1);
            var newEmb = NN.Embedding(num_embeddings: newRows, embedding_dims: _dim);
            
            newEmb.to(_device);

            using var _ = no_grad();

            var src = _emb!.weight.detach();
            var dst = newEmb.weight.detach();

            if (_rows > 0)
            {
                dst.narrow(dim: 0, start: 0, length: _rows)
                   .copy_(src.narrow(dim: 0, start: 0, length: _rows).to(_device));
            }

            _emb.Dispose();

            _emb = newEmb;
            _rows = newRows;

            _log.LogInformation("Expanded embedding rows: {Old} -> {New}", src.shape[0], newRows);
        }

        public float[] Embed(int[] tokenIds)
        {
            if (tokenIds == null || tokenIds.Length == 0) 
                return new float[_dim];

            var maxId = tokenIds.Max();

            EnsureRows(maxId);


            using var t = tensor(tokenIds.Select(i => (long)i).ToArray(), dtype: ScalarType.Int64, device: _device);
            using var e = _emb!.forward(t);
            using var v = e.mean([0L]);
            using var n = v / (v.norm() + 1e-9);

            var cpu = n.to(type: ScalarType.Float32, device: CPU);
            var result = new float[_dim];
            
            cpu.data<float>().CopyTo(result);

            return result;
        }

        public (int epochs, double finalLoss) Train((int[] A, int[] B, int Label)[] pairs)
        {
            if (pairs == null || pairs.Length == 0) 
                return (0, 0);

            var cfg = _opts.CurrentValue;

            random.manual_seed(cfg.Seed);

            var maxToken = pairs.SelectMany(p => p.A.Concat(p.B)).DefaultIfEmpty(0).Max();
            
            EnsureRows(maxToken);

            using var optimizer = OPT.Adam(_emb!.parameters(), lr: cfg.LearningRate);
            
            double lastLoss = 0;

            for (int epoch = 0; epoch < cfg.Epochs; epoch++)
            {
                lastLoss = 0;

                var order = Enumerable.Range(0, pairs.Length).OrderBy(_ => Guid.NewGuid()).ToArray();

                for (int i = 0; i < order.Length; i += cfg.BatchSize)
                {
                    var batchIdx = order.Skip(i).Take(cfg.BatchSize).ToArray();
                    
                    if (batchIdx.Length == 0) 
                        break;

                    using var aEmb = BatchEmbed(pairs, batchIdx, pickA: true);
                    using var pEmb = BatchEmbed(pairs, batchIdx, pickA: false);
                    using var nEmb = pEmb.roll(shifts: 1, dims: 0);

                    _log.LogInformation("aEmb={A} pEmb={P}", string.Join('x', aEmb.shape), string.Join('x', pEmb.shape));

                    using var simPos = F.cosine_similarity(aEmb, pEmb, dim: 1, eps: 1e-8);
                    using var simNeg = F.cosine_similarity(aEmb, nEmb, dim: 1, eps: 1e-8);

                    using var margin = simPos - simNeg;
                    using var loss = F.softplus(-margin).mean();

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();

                    lastLoss += loss.ToDouble();
                }

                _log.LogInformation("Torch epoch {Epoch}/{Total} loss={Loss:F4}", epoch + 1, cfg.Epochs, lastLoss);
            }

            return (cfg.Epochs, lastLoss);
        }

        private Tensor BatchEmbed((int[] A, int[] B, int Label)[] pairs, int[] batchIdx, bool pickA)
        {
            var list = new Tensor[batchIdx.Length];

            for (int r = 0; r < batchIdx.Length; r++)
            {
                var ids = pickA ? pairs[batchIdx[r]].A : pairs[batchIdx[r]].B;
                var maxId = ids.Length == 0 ? 0 : ids.Max();

                EnsureRows(maxId);
                
                var t = tensor(ids.Select(i => (long)i).ToArray(), dtype: ScalarType.Int64, device: _device);
                var e = _emb!.forward(t);                                       
                var v = e.mean([0L]);
                var n = v / (v.norm() + 1e-9);                                  

                t.Dispose();
                e.Dispose();
                v.Dispose();

                list[r] = n; 
            }

            var batch = stack(list, dim: 0);

            return batch;
        }

        public void Save()
        {
            using var _ = no_grad();

            var w = _emb!.weight.detach().cpu().to_type(ScalarType.Float32);
            var arr = new float[_rows * _dim];

            w.data<float>().CopyTo(arr);

            _store.SaveMatrix("emb", arr, _rows, _dim);
        }

        public void Dispose()
            => _emb?.Dispose();    
    }
}
