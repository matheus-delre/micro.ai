using Core.Models;

namespace Core.Abstractions
{
    public interface IStateStore
    {
        bool TryLoad(out State state);
        void Save(State state);
    }
}
