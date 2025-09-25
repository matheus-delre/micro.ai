namespace Core.Options
{
    public sealed class TokenizerOptions
    {
        public bool Lowercase { get; set; } = true;
        public int EncodeCacheSize { get; set; } = 2048;
        public int DecodeCacheSize { get; set; } = 2048;
        public GarbledPolicy GarbledPolicy { get; set; } = GarbledPolicy.Replace;
        public double GarbledThreshold { get; set; } = 0.02;
        public SpecialIds SpecialTokens { get; set; } = new();
        public int MinPairCount { get; set; } = 2; 
        public string Normalize { get; set; } = "NFKC";

        public sealed class SpecialIds
        {
            public int PAD { get; set; } = 0;
            public int BOS { get; set; } = 1;
            public int EOS { get; set; } = 2;
            public int MASK { get; set; } = 3;
            public int SEP { get; set; } = 4;
            public int UNK { get; set; } = 5;
            public int GARBLE { get; set; } = 6;
        }
    }

    public enum GarbledPolicy 
    { 
        Replace, 
        Keep, 
        Skip 
    }
}
