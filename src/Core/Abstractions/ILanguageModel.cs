namespace Core.Abstractions
{
    public interface ILanguageModel
    {
        int[] Generate(int[] promptIds, int maxNewTokens, float temperature = 1.0f, int topK = 0, float topP = 0.0f);
        
        (int epochs, double finalLoss) TrainSelfSupervised(IEnumerable<int> tokenStream, int? epochs = null);

        (int epochs, double finalLoss) TrainSFT(IEnumerable<(string Prompt, string Answer)> pairs, int? epochs = null);
        
        void Save();
    }
}
