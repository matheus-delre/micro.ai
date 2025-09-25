namespace Application.LLM
{
    public sealed class PackedDataset
    {
        private readonly int[] _tokens;
        private readonly Random _rng;

        public PackedDataset(IEnumerable<int> tokenStream, int seed = 42)
        {
            _tokens = [.. tokenStream ?? []];
            _rng = new Random(seed);
            
            if (_tokens.Length < 2)
                throw new ArgumentException("Token stream must contain at least 2 tokens.");
        }

        public IEnumerable<(int[] X, int[] Y)> Batches(int batchSize, int seqLen, int epochs, bool shuffle = true)
        {
            var N = _tokens.Length - 1;
            var starts = new List<int>();
            
            for (int i = 0; i + seqLen < N; i += seqLen) 
                starts.Add(i);

            for (int e = 0; e < epochs; e++)
            {
                if (shuffle)
                    Shuffle(starts);

                for (int p = 0; p < starts.Count; p += batchSize)
                {
                    var curB = Math.Min(batchSize, starts.Count - p);

                    if (curB <= 0) 
                        break;

                    var flatX = new int[curB * seqLen];
                    var flatY = new int[curB * seqLen];

                    for (int b = 0; b < curB; b++)
                    {
                        var s = starts[p + b];
                        var offX = b * seqLen;

                        Array.Copy(_tokens, s, flatX, offX, seqLen);
                        Array.Copy(_tokens, s + 1, flatY, offX, seqLen);
                    }

                    yield return (flatX, flatY);
                }
            }
        }

        private void Shuffle(List<int> list)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = _rng.Next(i + 1);

                (list[i], list[j]) = (list[j], list[i]);
            }
        }
    }
}
