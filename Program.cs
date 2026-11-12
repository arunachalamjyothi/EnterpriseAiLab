using System.Text.Json;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================");
Console.WriteLine("  Day 10: Reciprocal Rank Fusion (RRF) & Candidate Merging       ");
Console.WriteLine("================================================================\n");
Console.ResetColor();

// 1. Simulate dense vector retrieval results (Bounded Cosine: 0.0 to 1.0)
var vectorResults = new List<SearchResult>
{
    new("DOC-102", "Health insurance reimbursement guidelines", 0.89f),
    new("DOC-305", "Dental and optical claim limits", 0.74f),
    new("DOC-204", "Hospital invoice and discharge submission checklist", 0.71f),
    new("DOC-408", "Corporate laptop security encryption standard", 0.52f)
};

// 2. Simulate sparse BM25 lexical results (Unbounded term score: 0 to 50+)
var keywordResults = new List<SearchResult>
{
    new("DOC-501", "Accidental injury claim forms and guidelines", 28.4f),
    new("DOC-204", "Hospital invoice and discharge submission checklist", 22.1f),
    new("DOC-102", "Health insurance reimbursement guidelines", 18.6f),
    new("DOC-909", "Overseas corporate credit card settlement policy", 9.5f)
};

Console.WriteLine("--- 1. Raw Input: Dense Vector Search (Ranked) ---");
for (int i = 0; i < vectorResults.Count; i++)
{
    Console.WriteLine($"  Rank #{i + 1} | Score: {vectorResults[i].RawScore:F4} | ID: {vectorResults[i].DocId} | \"{vectorResults[i].Title}\"");
}

Console.WriteLine("\n--- 2. Raw Input: Sparse BM25 Keyword Search (Ranked) ---");
for (int i = 0; i < keywordResults.Count; i++)
{
    Console.WriteLine($"  Rank #{i + 1} | Score: {keywordResults[i].RawScore:F1} | ID: {keywordResults[i].DocId} | \"{keywordResults[i].Title}\"");
}

// 3. Execute Reciprocal Rank Fusion (k = 60)
var fusedResults = RrfMerger.Merge(vectorResults, keywordResults, k: 60);

// 4. Output Unified Ranked Results
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("\n================================================================");
Console.WriteLine("  Unified Candidates via Reciprocal Rank Fusion (k = 60)         ");
Console.WriteLine("================================================================");
Console.ResetColor();

int finalRank = 1;
foreach (var item in fusedResults)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write($"Final Rank #{finalRank++} ");
    Console.ResetColor();
    Console.WriteLine($"| RRF Score: {item.RrfScore:F5} | Doc ID: {item.DocId}");
    
    string vecRankStr = item.VectorRank.HasValue ? $"#{item.VectorRank.Value}" : "N/A";
    string bm25RankStr = item.KeywordRank.HasValue ? $"#{item.KeywordRank.Value}" : "N/A";
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"   [Retriever Agreement] Vector Rank: {vecRankStr} | Keyword Rank: {bm25RankStr}");
    Console.ResetColor();
    Console.WriteLine($"   Content: \"{item.Title}\"\n");
}

// -------------------------------------------------------------
// Core Domain Models & RRF Merger Utility
// -------------------------------------------------------------
public record SearchResult(string DocId, string Title, float RawScore);

public record FusedCandidate(
    string DocId,
    string Title,
    double RrfScore,
    int? VectorRank,
    int? KeywordRank
);

public static class RrfMerger
{
    public static List<FusedCandidate> Merge(
        List<SearchResult> vectorList,
        List<SearchResult> keywordList,
        int k = 60)
    {
        // Dictionary tracking cumulative score and ranks per document
        var candidateMap = new Dictionary<string, (string Title, double Score, int? VecRank, int? KeyRank)>();

        // Process Vector Retriever (1-based rank indexing)
        for (int i = 0; i < vectorList.Count; i++)
        {
            var item = vectorList[i];
            int rank = i + 1;
            double reciprocalScore = 1.0 / (k + rank);

            candidateMap[item.DocId] = (item.Title, reciprocalScore, rank, null);
        }

        // Process Keyword Retriever and fuse scores
        for (int i = 0; i < keywordList.Count; i++)
        {
            var item = keywordList[i];
            int rank = i + 1;
            double reciprocalScore = 1.0 / (k + rank);

            if (candidateMap.TryGetValue(item.DocId, out var existing))
            {
                // Document present in both retrievers: Sum reciprocal ranks
                candidateMap[item.DocId] = (
                    existing.Title,
                    existing.Score + reciprocalScore,
                    existing.VecRank,
                    rank
                );
            }
            else
            {
                // Document exclusive to keyword search
                candidateMap[item.DocId] = (item.Title, reciprocalScore, null, rank);
            }
        }

        // Project and sort descending by total RRF score
        return candidateMap
            .Select(kvp => new FusedCandidate(
                DocId: kvp.Key,
                Title: kvp.Value.Title,
                RrfScore: kvp.Value.Score,
                VectorRank: kvp.Value.VecRank,
                KeywordRank: kvp.Value.KeyRank
            ))
            .OrderByDescending(c => c.RrfScore)
            .ToList();
    }
}
