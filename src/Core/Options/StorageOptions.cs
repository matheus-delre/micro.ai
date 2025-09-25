namespace Core.Options
{
    public sealed class StorageOptions
    {
        public string Provider { get; set; } = "Rocks";
        public string BasePath { get; set; } = "state";
        public string BpeStateFile { get; set; } = "bpe_state.json";
    }
}
