using Core.Models;

namespace Core.Abstractions
{
    public interface IDiagnostics
    {
        (int applied, List<PairCount> appliedPairs, List<PairCount> topPairsFinal)
            SimulateTrain(IEnumerable<string> corpus, int numMerges, int minPairCount, bool lowercase, string normalize);

        List<PairCount> TopPairs(IEnumerable<string> corpus, int take, bool lowercase, string normalize);
    }
}
