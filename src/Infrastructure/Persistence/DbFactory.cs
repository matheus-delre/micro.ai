using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using RocksDbSharp;

namespace Infrastructure.Persistence
{
    public sealed class DbFactory : IDisposable
    {
        private readonly ILogger<DbFactory> _log;
        private readonly IOptionsMonitor<RocksDbOptions> _opts;
        private RocksDb? _db;
        private readonly Dictionary<string, ColumnFamilyHandle> _cfs = new();

        public DbFactory(ILogger<DbFactory> log, IOptionsMonitor<RocksDbOptions> opts)
        {
            _log = log; 
            _opts = opts; 

            EnsureOpen(); 
        }

        public RocksDb Db => 
            _db ?? throw new InvalidOperationException("RocksDB is not opened.");

        public ColumnFamilyHandle CF(string name)
        {
            if (_cfs.TryGetValue(name, out var h)) 
                return h;

            throw new InvalidOperationException($"Column family '{name}' not found.");
        }

        public void EnsureOpen()
        {
            if (_db != null) 
                return;

            var cfg = _opts.CurrentValue;

            var basePath = Path.GetFullPath(cfg.BasePath ?? "state/rocks");
            Directory.CreateDirectory(basePath);

            var dbPath = Path.Combine(cfg.BasePath, "microai");

            var dbOpts = new DbOptions()
                .SetCreateIfMissing(true)
                .SetCreateMissingColumnFamilies(true)
                .SetMaxOpenFiles(cfg.MaxOpenFiles);

            if (cfg.EnableStatistics)
                dbOpts.EnableStatistics();

            string[] onDisk;

            try
            {
                onDisk = RocksDb.ListColumnFamilies(new DbOptions(), dbPath)?.ToArray()
                         ?? [];
            }
            catch
            {
                onDisk = [];
            }

            var desired = new[] { "default", "bpe", "docs", "postings", "vectors", "nn", "meta" };

            var toOpen = onDisk.Length > 0
                ? [.. onDisk.Union(desired, StringComparer.Ordinal).Distinct()]
                : desired;

            var families = new ColumnFamilies();

            foreach (var name in toOpen)
                families.Add(name, new ColumnFamilyOptions());

            _db = RocksDb.Open(dbOpts, dbPath, families);

            _cfs.Clear();
            _cfs["default"] = _db.GetDefaultColumnFamily();

            foreach (var name in toOpen.Where(n => n != "default"))
                _cfs[name] = _db.GetColumnFamily(name);

            _log.LogInformation("RocksDB opened at {Path} with CFs: {CFs}",
                dbPath, string.Join(",", _cfs.Keys));
        }

        public void Dispose()
        {
            _db?.Dispose();
        }
    }
}
