using System.Diagnostics;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

const string collectionName = "enterprise_knowledge_base";
const ulong vectorSize = 768; // nomic-embed-text dimension

// 1. Plumbing setup
var qdrantClient = new QdrantClient("localhost", 6334);
var httpClient = new HttpClient { BaseAddress = new Uri("http://localhost:11434/v1") };

var builder = Kernel.CreateBuilder();
builder.AddOpenAITextEmbeddingGeneration(
    modelId: "nomic-embed-text",
    apiKey: "local-ollama",
    httpClient: httpClient
);

Kernel kernel = builder.Build();

#pragma warning disable CS0618
var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
#pragma warning restore CS0618

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================");
Console.WriteLine("  Day 6: End-to-End Semantic Pipeline with Latency Profiler     ");
Console.WriteLine("================================================================\n");
Console.ResetColor();

// 2. Fresh collection creation
if (await qdrantClient.CollectionExistsAsync(collectionName))
{
    await qdrantClient.DeleteCollectionAsync(collectionName);
}

await qdrantClient.CreateCollectionAsync(
    collectionName: collectionName,
    vectorsConfig: new VectorParams { Size = vectorSize, Distance = Distance.Cosine }
);

// 3. Ingestion paragraph
string sourceDocument = 
    "Employees traveling for international on-site client assignments are entitled to a daily " +
    "per-diem allowance of $75 for meals and incidental expenses. All original hotel tax receipts " +
    "and boarding passes must be uploaded to the Oracle ERP expense portal within 15 calendar days " +
    "of returning to the home base location. Expenses exceeding the threshold require Director approval.";

Console.WriteLine("Ingesting unstructured enterprise policy paragraph into Qdrant...\n");

var ingestStopwatch = Stopwatch.StartNew();
var docEmbeddings = await embeddingService.GenerateEmbeddingsAsync([sourceDocument], kernel: kernel);
ingestStopwatch.Stop();

var point = new PointStruct
{
    Id = 1UL,
    Vectors = docEmbeddings[0].ToArray()
};
point.Payload.Add("docId", "DOC-EXP-2026-INTL");
point.Payload.Add("category", "Travel & Expense");
point.Payload.Add("content", sourceDocument);

await qdrantClient.UpsertAsync(collectionName, [point]);

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"[Document Indexed Successfully]");
Console.WriteLine($"Ingestion Embedding Time (T1_ingest): {ingestStopwatch.ElapsedMilliseconds} ms\n");
Console.ResetColor();

// 4. Interactive retrieval loop with latency profiling
Console.WriteLine("Type a natural language query (or type 'exit' to quit):");
Console.WriteLine("Sample: 'How much food allowance do I get on international client trips?'\n");

while (true)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write("\nQuery > ");
    Console.ResetColor();
    
    string? query = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(query) || query.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
    {
        break;
    }

    // T1: Time to embed query text
    var swEmbed = Stopwatch.StartNew();
    var queryEmbeddings = await embeddingService.GenerateEmbeddingsAsync([query], kernel: kernel);
    swEmbed.Stop();
    long t1EmbedMs = swEmbed.ElapsedMilliseconds;

    // T2: Time to search HNSW index
    var swSearch = Stopwatch.StartNew();
#pragma warning disable CS0618
    var searchResults = await qdrantClient.SearchAsync(
        collectionName: collectionName,
        vector: queryEmbeddings[0].ToArray(),
        limit: 1
    );
#pragma warning restore CS0618
    swSearch.Stop();
    long t2SearchMs = swSearch.ElapsedMilliseconds;

    // Display telemetry & top match
    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.WriteLine($"\n[Latency Telemetry] Embedding (T1): {t1EmbedMs} ms | HNSW Search (T2): {t2SearchMs} ms | Total: {t1EmbedMs + t2SearchMs} ms");
    Console.ResetColor();

    foreach (var hit in searchResults)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"Match Score   : {hit.Score:F4} (Cosine Similarity)");
        Console.WriteLine($"Document ID   : {hit.Payload["docId"].StringValue}");
        Console.WriteLine($"Category      : {hit.Payload["category"].StringValue}");
        Console.WriteLine($"Retrieved Text: {hit.Payload["content"].StringValue}");
        Console.ResetColor();
    }
}
