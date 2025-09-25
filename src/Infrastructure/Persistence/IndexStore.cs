using Core.Models;
using System.Text;
using System.Text.Json;

namespace Infrastructure.Persistence
{
    public sealed class IndexStore(DbFactory factory)
    {
        private static readonly Encoding U = Encoding.UTF8;

        public void UpsertDocument(Document doc)
        {
            var db = factory.Db; 
            var cf = factory.CF("docs");

            var key = U.GetBytes($"doc:{doc.Id}");

            db.Put(key, JsonSerializer.SerializeToUtf8Bytes(doc), cf);
        }

        public Document? GetDocument(string id)
        {
            var db = factory.Db; 
            var cf = factory.CF("docs");

            var val = db.Get($"doc:{id}", cf);

            return val == null ? null : JsonSerializer.Deserialize<Document>(val);
        }

        public void UpdatePostings(IEnumerable<int> tokenIds, string docId)
        {
            var db = factory.Db; var cf = factory.CF("postings");

            foreach (var g in tokenIds.GroupBy(x => x))
            {
                var key = U.GetBytes($"t:{g.Key}");
                var existing = db.Get(key, cf);

                var map = existing == null ? [] : JsonSerializer.Deserialize<Dictionary<string, int>>(existing)!;
                
                map[docId] = g.Count();

                db.Put(key, JsonSerializer.SerializeToUtf8Bytes(map), cf);
            }
        }

        public Dictionary<string, int> GetPosting(int term)
        {
            var db = factory.Db; 
            var cf = factory.CF("postings");

            var val = db.Get($"t:{term}", cf);

            return val == null ? [] : JsonSerializer.Deserialize<Dictionary<string, int>>(val)!;
        }

        public IEnumerable<Document> EnumerateDocuments()
        {
            var db = factory.Db;
            var cf = factory.CF("docs");
            var it = db.NewIterator(cf);

            var prefix = Encoding.UTF8.GetBytes("doc:");
 
            for (it.SeekToFirst(); it.Valid(); it.Next())
            {
                var k = it.Key(); 

                if (k == null || k.Length < prefix.Length) 
                    continue;

                bool hasPrefix = true;

                for (int i = 0; i < prefix.Length; i++)
                    if (k[i] != prefix[i]) 
                    { 
                        hasPrefix = false; 
                        break; 
                    }

                if (!hasPrefix) 
                    continue;

                var id = Encoding.UTF8.GetString(k.AsSpan(prefix.Length));

                var val = it.Value();

                if (val == null) 
                    continue;

                Document? parsed = null;
                try 
                { 
                    parsed = JsonSerializer.Deserialize<Document>(val); 
                }
                catch 
                { 
                }

                if (parsed == null || string.IsNullOrWhiteSpace(parsed.Id))
                {
                    yield return new Document
                    {
                        Id = id,
                        Text = parsed?.Text ?? string.Empty,
                        Metadata = parsed?.Metadata ?? default
                    };
                }
                else
                {
                    yield return parsed;
                }
            }

            it.Dispose();
        }
    }
}
