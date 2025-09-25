using Core.Abstractions;
using Core.Models;
using Microsoft.Extensions.Logging;
using RocksDbSharp;
using System.Text;
using System.Text.Json;
namespace Infrastructure.Persistence
{
    public sealed class CheckpointStore(ILogger<CheckpointStore> log, DbFactory factory) : ICheckpointStore
    {
        private readonly ILogger<CheckpointStore> _log = log;

        private ColumnFamilyHandle CF() 
            => factory.CF("nn");

        private static byte[] K(string s) 
            => Encoding.UTF8.GetBytes(s);

        public void SaveNamed(string prefix, IReadOnlyList<CheckpointTensor> tensors, object? optionsSnapshot = null)
        {
            var db = factory.Db; 
            var cf = CF();

            var shapes = tensors.Select(t => t.Shape).ToList();
            var names = tensors.Select(t => t.Name).ToList();

            var meta = new CheckpointMeta { Version = 1, Schema = "named-tensors", Names = names, Shapes = shapes };

            db.Put(K($"{prefix}:meta:shapes"), JsonSerializer.SerializeToUtf8Bytes(shapes), cf);
            db.Put(K($"{prefix}:meta:names"), JsonSerializer.SerializeToUtf8Bytes(names), cf);
            db.Put(K($"{prefix}:meta"), JsonSerializer.SerializeToUtf8Bytes(meta), cf);

            if (optionsSnapshot is not null)
                db.Put(K($"{prefix}:opts"), JsonSerializer.SerializeToUtf8Bytes(optionsSnapshot), cf);

            for (int i = 0; i < tensors.Count; i++)
            {
                var t = tensors[i];
                var bytes = new byte[t.Data.Length * sizeof(float)];

                Buffer.BlockCopy(t.Data, 0, bytes, 0, bytes.Length);

                db.Put(K($"{prefix}:param:{i:D4}"), bytes, cf);
            }

            _log.LogInformation("Checkpoint[{Prefix}] saved: {Count} tensors", prefix, tensors.Count);
        }

        public (IReadOnlyList<CheckpointTensor> Tensors, CheckpointMeta Meta)? LoadNamed(string prefix)
        {
            var db = factory.Db; var cf = CF();

            var namesBytes = db.Get(K($"{prefix}:meta:names"), cf);
            var shapesBytes = db.Get(K($"{prefix}:meta:shapes"), cf);

            if (namesBytes is null || shapesBytes is null)
                return null;

            List<string>? names;
            List<long[]>? shapes;

            try
            {
                names = JsonSerializer.Deserialize<List<string>>(namesBytes);
                shapes = JsonSerializer.Deserialize<List<long[]>>(shapesBytes);
            }
            catch
            {
                return null;
            }

            if (names is null || shapes is null || names.Count != shapes.Count)
                return null;

            var list = new List<CheckpointTensor>(names.Count);

            for (int i = 0; i < names.Count; i++)
            {
                var dataBytes = db.Get(K($"{prefix}:param:{i:D4}"), cf);

                if (dataBytes is null)
                    return null;

                var floats = new float[dataBytes.Length / sizeof(float)];

                Buffer.BlockCopy(dataBytes, 0, floats, 0, dataBytes.Length);

                list.Add(new CheckpointTensor(names[i], floats, shapes[i]));
            }

            CheckpointMeta meta;

            var metaBytes = db.Get(K($"{prefix}:meta"), cf);

            if (metaBytes is not null)
            {
                try 
                { 
                    meta = JsonSerializer.Deserialize<CheckpointMeta>(metaBytes) ?? new(); 
                }
                catch 
                { 
                    meta = new(); 
                }
            }
            else 
                meta = new();

            _log.LogInformation("Checkpoint[{Prefix}] loaded: {Count} tensors", prefix, list.Count);
            
            return (list, meta);
        }

        public void SaveMatrix(string prefix, float[] weights, int rows, int dim, object? optionsSnapshot = null)
        {
            SaveNamed(prefix,
            [
                new CheckpointTensor("weight", weights, [rows, dim])
            ], optionsSnapshot);
        }

        public (float[] Weights, int Rows, int Dim)? LoadMatrix(string prefix)
        {
            var snap = LoadNamed(prefix);

            if (snap is null) 
                return null;

            var t = snap.Value.Tensors.FirstOrDefault();

            if (t is null || t.Shape.Length != 2)
                return null;

            return (t.Data, (int)t.Shape[0], (int)t.Shape[1]);
        }

        public void SaveOptions<T>(string prefix, T options)
        {
            var db = factory.Db; 
            var cf = CF();

            db.Put(K($"{prefix}:opts"), JsonSerializer.SerializeToUtf8Bytes(options!), cf);
        }

        public T? LoadOptions<T>(string prefix)
        {
            var db = factory.Db; 
            var cf = CF();

            var bytes = db.Get(K($"{prefix}:opts"), cf);

            if (bytes is null) 
                return default;

            try 
            { 
                return JsonSerializer.Deserialize<T>(bytes); 
            }
            catch 
            {
                return default; 
            }
        }
    }
}