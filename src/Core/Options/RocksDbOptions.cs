namespace Core.Options
{
    public sealed class RocksDbOptions
    {
        public string BasePath { get; set; } = "state/rocks";
        public bool EnableStatistics { get; set; } = true;
        public int BlockCacheMB { get; set; } = 512;
        public int MaxOpenFiles { get; set; } = 512;
        public bool UseBloomFilter { get; set; } = true;
    }
}
