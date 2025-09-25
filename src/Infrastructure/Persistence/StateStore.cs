using Core.Abstractions;
using Core.Models;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using RocksDbSharp;
using System.Text;
using System.Text.Json;

namespace Infrastructure.Persistence
{
    public sealed class StateStore(
        ILogger<StateStore> log,
        IOptionsMonitor<StorageOptions> storage,
        DbFactory factory) : IStateStore
    {
        private static readonly Encoding U = Encoding.UTF8;
        private static readonly byte[] BpeKey = U.GetBytes("bpe:state"); 

        public bool TryLoad(out State state)
        {
            var db = factory.Db;
            var cf = factory.CF("bpe");

            var val = db.Get(BpeKey, cf);

            if (val is not null)
            {
                state = JsonSerializer.Deserialize<State>(val)!;

                log.LogInformation("Loaded BPE state from RocksDB.");

                return true;
            }

            try
            {
                var basePath = storage.CurrentValue.BasePath;
                var path = Path.Combine(basePath, storage.CurrentValue.BpeStateFile);

                if (File.Exists(path))
                {
                    var json = File.ReadAllText(path);
                    state = JsonSerializer.Deserialize<State>(json)!;

                    Save(state);

                    log.LogInformation("Imported legacy BPE state from file: {Path}", path);
                    
                    return true;
                }
            }
            catch (Exception ex)
            {
                log.LogWarning(ex, "Failed to import legacy BPE state file.");
            }

            state = null!;

            return false;
        }

        public void Save(State state)
        {
            var db = factory.Db;
            var cf = factory.CF("bpe");

            var json = JsonSerializer.SerializeToUtf8Bytes(state, new JsonSerializerOptions { WriteIndented = false });

            using var batch = new WriteBatch();

            batch.Put(BpeKey, json, cf);

            var wo = new WriteOptions().SetSync(true);

            db.Write(batch, wo);

            log.LogInformation("Saved BPE state to RocksDB (flushed + WAL synced).");
        }
    }
}
