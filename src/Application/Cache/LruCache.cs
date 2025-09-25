using System.Collections.Concurrent;

namespace Application.Cache
{
    public sealed class LruCache<TKey, TValue>(int capacity) where TKey : notnull
    {
        private readonly int _capacity = capacity <= 0 ? 1 : capacity;
        private readonly LinkedList<(TKey Key, TValue Value)> _list = new();
        private readonly ConcurrentDictionary<TKey, LinkedListNode<(TKey key, TValue val)>> _map = new();
        private readonly LinkedList<(TKey key, TValue val)> _lru = new();
        private readonly Lock _gate = new();

        public bool TryGet(TKey key, out TValue value)
        {
            if (_map.TryGetValue(key, out var node))
            {
                lock (_gate)
                {
                    _lru.Remove(node);
                    _lru.AddFirst(node);
                }

                value = node.Value.val;

                return true;
            }

            value = default;

            return false;
        }

        public void Set(TKey key, TValue value)
        {
            lock (_gate)
            {
                if (_map.TryGetValue(key, out var existing))
                {
                    existing.Value = (key, value);
                    _lru.Remove(existing);
                    _lru.AddFirst(existing);

                    return;
                }

                var node = new LinkedListNode<(TKey, TValue)>((key, value));

                _lru.AddFirst(node);
                _map[key] = node;

                if (_map.Count > _capacity)
                {
                    var last = _lru.Last;

                    if (last != null)
                    {
                        _lru.RemoveLast();
                        _map.TryRemove(last.Value.key, out _);
                    }
                }
            }
        }

        public void Clear()
        {
            lock (_gate)
            {
                _map.Clear();
                _list.Clear();
            }
        }

        public int Count 
            => _map.Count;
    }
}
