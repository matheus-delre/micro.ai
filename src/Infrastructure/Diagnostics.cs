using Core.Abstractions;
using Core.Models;
using Microsoft.Extensions.Logging;
using System.Globalization;
using System.Text;

namespace Infrastructure
{
    public sealed class Diagnostics(ILogger<Diagnostics> log) : IDiagnostics
    {
        public (int applied, List<PairCount> appliedPairs, List<PairCount> topPairsFinal)
            SimulateTrain(IEnumerable<string> corpus, int numMerges, int minPairCount, bool lowercase, string normalize)
        {
            var words = PrepareWords(corpus, lowercase, normalize)
                        .Select(w => w.Select(c => c.ToString()).ToList())
                        .Where(lst => lst.Count > 0)
                        .ToList();

            var applied = 0;
            var appliedPairs = new List<PairCount>();

            while (applied < numMerges)
            {
                var freq = CountPairs(words);

                if (freq.Count == 0) 
                    break;

                var (left, right, count) = freq[0];

                if (count < minPairCount) 
                    break;

                ApplyMerge(words, left, right);

                applied++;

                appliedPairs.Add(new PairCount(left, right, count));
            }

            var finalTop = CountPairs(words, take: 50).Select(t => new PairCount(t.Left, t.Right, t.Count)).ToList();

            log.LogInformation("BPE dry-run: applied={Applied}, lastPair={Last}", applied, appliedPairs.LastOrDefault());

            return (applied, appliedPairs, finalTop);
        }

        public List<PairCount> TopPairs(IEnumerable<string> corpus, int take, bool lowercase, string normalize)
        {
            var words = PrepareWords(corpus, lowercase, normalize)
                        .Select(w => w.Select(c => c.ToString()).ToList())
                        .Where(lst => lst.Count > 1)
                        .ToList();

            return [.. CountPairs(words, take).Select(t => new PairCount(t.Left, t.Right, t.Count))];
        }

        private static IEnumerable<string> PrepareWords(IEnumerable<string> corpus, bool lowercase, string normalize)
        {
            foreach (var line in corpus ?? Array.Empty<string>())
            {
                var s = line ?? string.Empty;

                if (!string.IsNullOrEmpty(normalize))
                {
                    var form = normalize.ToUpperInvariant() switch
                    {
                        "NFC" => NormalizationForm.FormC,
                        "NFD" => NormalizationForm.FormD,
                        "NFKC" => NormalizationForm.FormKC,
                        "NFKD" => NormalizationForm.FormKD,
                        _ => NormalizationForm.FormKC
                    };

                    s = s.Normalize(form);
                }

                if (lowercase) 
                    s = s.ToLower(CultureInfo.InvariantCulture);

                foreach (var w in s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
                    yield return w.Trim();
            }
        }

        private static List<(string Left, string Right, int Count)> CountPairs(List<List<string>> words, int take = 1000)
        {
            var dict = new Dictionary<(string L, string R), int>();

            foreach (var w in words)
            {
                for (int i = 0; i < w.Count - 1; i++)
                {
                    var k = (w[i], w[i + 1]);

                    dict[k] = dict.TryGetValue(k, out var c) ? c + 1 : 1;
                }
            }

            return dict.Count == 0
                ? []
                : dict.Select(kv => (kv.Key.L, kv.Key.R, kv.Value))
                      .OrderByDescending(t => t.Value)
                      .ThenBy(t => t.L).ThenBy(t => t.R)
                      .Take(take)
                      .ToList();
        }

        private static void ApplyMerge(List<List<string>> words, string left, string right)
        {
            var merged = left + right;

            foreach (var w in words)
            {
                if (w.Count < 2) 
                    continue;

                var i = 0;

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
        }
    }
}
