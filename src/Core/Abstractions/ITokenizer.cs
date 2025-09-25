using Core.Models;

namespace Core.Abstractions
{
    public interface ITokenizer
    {
        int[] Encode(string text, bool addBosEos = false);

        string Decode(IEnumerable<int> ids);

        int Train(IEnumerable<string> corpus, int numMerges);

        IReadOnlyDictionary<string, int> GetVocabulary();

        IReadOnlyList<(string Left, string Right, string Merge)> GetMerges();

        IReadOnlyDictionary<string, int> GetSpecialTokenMap();

        State Snapshot();

        void Restore(State state);

        Metrics GetMetrics();
    }
}
