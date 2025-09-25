using Application.Cache;
using Core.Abstractions;
using Core.Enums;
using Core.Models;
using Core.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.RegularExpressions;

namespace Application
{
    public sealed class Tokenizer : ITokenizer
    {
        private readonly ILogger<Tokenizer> _log;
        private readonly IOptionsMonitor<TokenizerOptions> _opts;
        private readonly IStateStore _store;

        private readonly ReaderWriterLockSlim _lock = new();
        private Dictionary<string, int> _vocab = [];
        private List<MergeRule> _merges = [];
        private Dictionary<string, int> _special = [];

        private readonly LruCache<string, int[]> _encodeCache;
        private readonly LruCache<string, string> _decodeCache;

        private static readonly Regex Spaces = new("\\s+", RegexOptions.Compiled);

        public Tokenizer(ILogger<Tokenizer> log, IOptionsMonitor<TokenizerOptions> opts, IStateStore store)
        {
            _log = log;
            _opts = opts;
            _store = store;

            _encodeCache = new LruCache<string, int[]>(opts.CurrentValue.EncodeCacheSize);
            _decodeCache = new LruCache<string, string>(opts.CurrentValue.DecodeCacheSize);

            LoadOrInitialize();
        }

        private void LoadOrInitialize()
        {
            if (_store.TryLoad(out var state))
            {
                _vocab = state.Vocabulary;
                _merges = state.Merges;
                _special = state.SpecialTokenMap;

                _log.LogInformation("BPE state loaded: vocab={V}, merges={M}", _vocab.Count, _merges.Count);
                
                return;
            }

            var o = _opts.CurrentValue;

            _special = new Dictionary<string, int>
            {
                [nameof(SpecialToken.PAD)] = o.SpecialTokens.PAD,
                [nameof(SpecialToken.BOS)] = o.SpecialTokens.BOS,
                [nameof(SpecialToken.EOS)] = o.SpecialTokens.EOS,
                [nameof(SpecialToken.MASK)] = o.SpecialTokens.MASK,
                [nameof(SpecialToken.SEP)] = o.SpecialTokens.SEP,
                [nameof(SpecialToken.UNK)] = o.SpecialTokens.UNK,
                [nameof(SpecialToken.GARBLE)] = o.SpecialTokens.GARBLE,
            };

            _vocab = [];

            foreach (var kv in _special)
                _vocab[kv.Key] = kv.Value;

            _merges = [];

            _log.LogInformation("Initialized new BPE state (empty vocab except specials)");
        }

        public int[] Encode(string text, bool addBosEos = false)
        {
            if (string.IsNullOrEmpty(text)) 
                return [];

            var key = addBosEos + "|" + text;

            if (_encodeCache.TryGet(key, out var cached))
                return cached;

            var tokens = new List<int>();
            var norm = Normalize(text);

            _lock.EnterReadLock();

            try
            {
                if (addBosEos) 
                    tokens.Add(_special[nameof(SpecialToken.BOS)]);

                foreach (var word in SplitWords(norm))
                {
                    tokens.AddRange(EncodeWord(word));
                    tokens.Add(_special[nameof(SpecialToken.SEP)]);
                }

                if (tokens.Count > 0) 
                    tokens.RemoveAt(tokens.Count - 1);

                if (addBosEos) 
                    tokens.Add(_special[nameof(SpecialToken.EOS)]);
            }
            finally 
            { 
                _lock.ExitReadLock(); 
            }

            var arr = tokens.ToArray();

            _encodeCache.Set(key, arr);

            return arr;
        }

        public string Decode(IEnumerable<int> ids)
        {
            var key = string.Join(',', ids);

            if (_decodeCache.TryGet(key, out var s)) 
                return s;

            var sb = new StringBuilder();

            _lock.EnterReadLock();

            try
            {
                var inv = _vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

                foreach (var id in ids)
                {
                    if (_special.ContainsValue(id))
                    {
                        if (id == _special[nameof(SpecialToken.SEP)])
                            sb.Append(' ');
                        else if (id == _special[nameof(SpecialToken.MASK)])
                            sb.Append("<mask>");

                        continue;
                    }

                    if (inv.TryGetValue(id, out var t)) 
                        sb.Append(t);
                    else 
                        sb.Append('<').Append(id).Append('>');
                }
            }
            finally 
            { 
                _lock.ExitReadLock(); 
            }

            var result = Spaces.Replace(sb.ToString(), " ").Trim();

            _decodeCache.Set(key, result);

            return result;
        }

        public int Train(IEnumerable<string> corpus, int numMerges)
        {
            if (numMerges <= 0) 
                return 0;

            _lock.EnterWriteLock();

            try
            {
                var words = new List<List<string>>();

                foreach (var line in corpus ?? [])
                {
                    var s = Normalize(line) ?? string.Empty;
                                                            
                    foreach (var w in s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
                    {
                        if (w.Length == 0) 
                            continue;

                        var tokens = new List<string>(w.Length);

                        foreach (var ch in w)
                        {
                            var t = ch.ToString();

                            tokens.Add(t);

                            if (!_vocab.ContainsKey(t))
                                _vocab[t] = NextId();
                        }

                        words.Add(tokens);
                    }
                }

                if (words.Count == 0) 
                    return 0;

                var mergesAdded = 0;

                int minPairCount = _opts.CurrentValue.MinPairCount;

                while (mergesAdded < numMerges)
                {
                    var counts = new Dictionary<(string L, string R), int>();

                    foreach (var w in words)
                    {
                        for (int i = 0; i < w.Count - 1; i++)
                        {
                            var key = (w[i], w[i + 1]);

                            if (counts.TryGetValue(key, out var c)) 
                                counts[key] = c + 1;
                            else 
                                counts[key] = 1;
                        }
                    }

                    if (counts.Count == 0) 
                        break;

                    (string L, string R, int C) best = default;
                    
                    var found = false;

                    foreach (var kv in counts)
                    {
                        var c = kv.Value;

                        if (!found || c > best.C)
                        {
                            best = (kv.Key.L, kv.Key.R, c);
                            found = true;
                        }
                    }

                    if (!found || best.C < minPairCount) 
                        break;

                    var left = best.L;
                    var right = best.R;
                    var merged = left + right;

                    if (!_vocab.ContainsKey(merged))
                        _vocab[merged] = NextId();

                    _merges.Add(new MergeRule { Left = left, Right = right, Merge = merged, Rank = _merges.Count });

                    foreach (var w in words)
                    {
                        if (w.Count < 2) 
                            continue;

                        int i = 0;

                        while (i < w.Count - 1)
                        {
                            if (w[i] == left && w[i + 1] == right)
                            {
                                w[i] = merged;
                                w.RemoveAt(i + 1);
                            }
                            else i++;
                        }
                    }

                    mergesAdded++;
                }

                _log.LogInformation("BPE trained: {Added} merges. Vocab={Vocab} Merges={M}",
                    mergesAdded, _vocab.Count, _merges.Count);

                return mergesAdded;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        public IReadOnlyDictionary<string, int> GetVocabulary()
        {
            _lock.EnterReadLock();

            try 
            { 
                return new Dictionary<string, int>(_vocab); 
            }
            finally 
            { 
                _lock.ExitReadLock(); 
            }
        }

        public IReadOnlyList<(string Left, string Right, string Merge)> GetMerges()
        {
            _lock.EnterReadLock();

            try 
            { 
                return [.. _merges.Select(m => (m.Left, m.Right, m.Merge))]; 
            }
            finally 
            { 
                _lock.ExitReadLock();
            }
        }

        public IReadOnlyDictionary<string, int> GetSpecialTokenMap() 
            => new Dictionary<string, int>(_special);

        public State Snapshot()
        {
            _lock.EnterReadLock();

            try
            {
                var vocabCopy = _vocab.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
                var mergesCopy = _merges.Select(m => new MergeRule { Left = m.Left, Right = m.Right, Merge = m.Merge, Rank = m.Rank }).ToList();
                var specialCopy = _special.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);

                return new State
                {
                    Vocabulary = vocabCopy,
                    Merges = mergesCopy,
                    SpecialTokenMap = specialCopy,
                };
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        public void Restore(State state)
        {
            ArgumentNullException.ThrowIfNull(state);

            _lock.EnterWriteLock();

            try
            {
                var vocab = state.Vocabulary ?? new Dictionary<string, int>(StringComparer.Ordinal);
                var merges = state.Merges ?? [];
                var specials = state.SpecialTokenMap ?? new Dictionary<string, int>(StringComparer.Ordinal);

                _vocab = vocab.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
                _merges = [.. merges.Select(m => new MergeRule { Left = m.Left, Right = m.Right, Merge = m.Merge, Rank = m.Rank })];
                _special = specials.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);

                foreach (var kv in _special)
                    if (!_vocab.ContainsKey(kv.Key))
                        _vocab[kv.Key] = kv.Value;

                _encodeCache.Clear();
                _decodeCache.Clear();

                _log.LogInformation("Tokenizer restored: vocab={V}, merges={M}, specials={S}",
                    _vocab.Count, _merges.Count, _special.Count);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        public Metrics GetMetrics()
        {
            _lock.EnterReadLock();

            try
            {
                return new Metrics
                {
                    VocabSize = _vocab.Count,
                    MergeCount = _merges.Count,
                    EncodeCacheSize = _encodeCache.Count,
                    DecodeCacheSize = _decodeCache.Count
                };
            }
            finally 
            { 
                _lock.ExitReadLock(); 
            }
        }

        private string Normalize(string text)
        {
            var t = _opts.CurrentValue.Lowercase ? text.ToLowerInvariant() : text;

            if (IsGarbled(t))
            {
                var policy = _opts.CurrentValue.GarbledPolicy;

                if (policy == GarbledPolicy.Replace) 
                    return nameof(SpecialToken.GARBLE);

                if (policy == GarbledPolicy.Skip) 
                    return string.Empty;
            }
            return t;
        }

        private bool IsGarbled(string t)
        {
            int weird = t.Count(c => char.IsControl(c) && !char.IsWhiteSpace(c));
            var ratio = (double)weird / Math.Max(1, t.Length);
            
            return ratio > _opts.CurrentValue.GarbledThreshold;
        }

        private IEnumerable<string> SplitWords(string t)
        {
            if (string.IsNullOrWhiteSpace(t)) 
                yield break;

            var sb = new StringBuilder();

            foreach (var ch in t)
            {
                if (char.IsLetterOrDigit(ch)) 
                    sb.Append(ch);
                else
                {
                    if (sb.Length > 0) 
                    { 
                        yield return sb.ToString(); 
                        sb.Clear(); 
                    }

                    yield return ch.ToString();
                }
            }

            if (sb.Length > 0) 
                yield return sb.ToString();
        }

        private IEnumerable<int> EncodeWord(string word)
        {
            if (string.IsNullOrEmpty(word)) 
                yield break;

            var seq = word.Select(ch => ch.ToString()).ToList();

            foreach (var rule in _merges.OrderBy(m => m.Rank))
            {
                for (int i = 0; i < seq.Count - 1; i++)
                {
                    if (seq[i] == rule.Left && seq[i + 1] == rule.Right)
                    {
                        seq[i] = rule.Merge;

                        seq.RemoveAt(i + 1);

                        i = Math.Max(-1, i - 2);
                    }
                }
            }

            foreach (var s in seq)
            {
                if (_vocab.TryGetValue(s, out var id)) 
                    yield return id;
                else 
                    yield return _special[nameof(SpecialToken.UNK)];
            }
        }

        private int NextId()
            => _vocab.Count == 0 ? 0 : _vocab.Values.Max() + 1;
    }
}
