using Core.Abstractions;
using System.Text;

namespace Infrastructure.Persistence
{
    public sealed class VectorStore(DbFactory factory) : IVectorStore
    {
        private static readonly Encoding U = Encoding.UTF8;

        public void Upsert(string id, ReadOnlySpan<float> vector)
        {
            var db = factory.Db; 
            var cf = factory.CF("vectors");
            var key = U.GetBytes($"v:{id}");
            var v = vector.ToArray();
            var buf = new byte[v.Length * sizeof(float)];

            Buffer.BlockCopy(v, 0, buf, 0, buf.Length);

            db.Put(key, buf, cf);
        }

        public (string id, float score)[] TopK(ReadOnlySpan<float> query, int k)
        {
            var db = factory.Db; var cf = factory.CF("vectors");
            var q = query.ToArray();
            var qnorm = MathF.Sqrt(q.Sum(x => x * x));
            var res = new List<(string id, float score)>();

            using var it = db.NewIterator(cf);

            it.SeekToFirst();

            while (it.Valid())
            {
                var keyBytes = it.Key();
                var key = U.GetString(keyBytes);

                if (key.StartsWith("v:"))
                {
                    var id = key.Substring(2);
                    var val = it.Value();
                    var vec = new float[val.Length / sizeof(float)];

                    Buffer.BlockCopy(val, 0, vec, 0, val.Length);

                    if (vec.Length == q.Length)
                    {
                        float dot = 0f, vnorm = 0f;

                        for (int i = 0; i < vec.Length; i++) 
                        { 
                            var vi = vec[i]; 
                            dot += vi * q[i]; 
                            vnorm += vi * vi; 
                        }

                        var score = (qnorm == 0 || vnorm == 0) ? 0f : dot / (qnorm * MathF.Sqrt(vnorm));
                        
                        res.Add((id, score));
                    }
                }

                it.Next();
            }

            return [.. res.OrderByDescending(x => x.score).Take(k)];
        }

        public bool TryGet(string id, out float[] vector)
        {
            var db = factory.Db; 
            var cf = factory.CF("vectors");
            var key = U.GetBytes($"v:{id}");
            var val = db.Get(key, cf); 

            if (val == null) 
            { 
                vector = []; 
                return false; 
            }

            var vec = new float[val.Length / sizeof(float)];

            Buffer.BlockCopy(val, 0, vec, 0, val.Length);

            vector = vec; return true;
        }

        public long Count()
        {
            var db = factory.Db; 
            var cf = factory.CF("vectors");

            long n = 0;

            using var it = db.NewIterator(cf);

            for (it.SeekToFirst(); it.Valid(); it.Next())
                if (U.GetString(it.Key()).StartsWith("v:"))
                    n++;

            return n;
        }
    }
}
