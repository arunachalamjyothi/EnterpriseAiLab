using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

const string collectionName = "enterprise_hybrid_policies";
const ulong vectorSize = 768; // nomic-embed-text dimension

// 1. Initialize Clients
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
Console.WriteLine("  Day 9: Hybrid Search Architecture (Dense Vector + BM25 Match) ");
Console.WriteLine("================================================================\n");
Console.ResetColor();

// 2. Create Collection Fresh
if (await qdrantClient.CollectionExistsAsync(collectionName))
{
    await qdrantClient.DeleteCollectionAsync(collectionName);
}

await qdrantClient.CreateCollectionAsync(
    collectionName: collectionName,
    vectorsConfig: new VectorParams { Size = vectorSize, Distance = Distance.Cosine }
);

// 3. Create Payload Full-Text Index for Keyword Matching (Lexical Indexing)
await qdrantClient.CreatePayloadIndexAsync(
    collectionName: collectionName,
    fieldName: "content",
    schemaType: PayloadSchemaType.Text,
    indexParams: new PayloadIndexParams
    {
        TextIndexParams = new TextIndexParams
        {
        Tokenizer = TokenizerType.Word, // Word tokenization splits on spaces and punctuation
        Lowercase = true
        }
    }
);
Console.WriteLine("[Created HNSW Dense Vector Index & Full-Text Lexical Payload Index]\n");

// 4. Ingest Documents with Similar Context but Different Alphanumeric IDs
var documents = new[]
{
    new
    {
        Id = 1UL,
        PolicyId = "POL-1002",
        Content = "Policy POL-1002 covers residential pipe bursts and plumbing leaks. The standard deductible is $250 for all residential water claims."
    },
    new
    {
        Id = 2UL,
        PolicyId = "POL-4819",
        Content = "Policy POL-4819 specifically governs commercial warehouse flood and flash water damage. The mandatory deductible is $2,500."
    },
    new
    {
        Id = 3UL,
        PolicyId = "POL-7730",
        Content = "Policy POL-7730 outlines corporate vehicle collision and storm hail damage deductible waivers."
    }
};

Console.WriteLine("Ingesting policy records with embeddings and payloads...");

var texts = documents.Select(d => d.Content).ToList();
var embeddings = await embeddingService.GenerateEmbeddingsAsync(texts, kernel: kernel);

var points = new List<PointStruct>();
for (int i = 0; i < documents.Length; i++)
{
    var pt = new PointStruct
    {
        Id = documents[i].Id,
        Vectors = embeddings[i].ToArray()
    };
    pt.Payload.Add("policyId", documents[i].PolicyId);
    pt.Payload.Add("content", documents[i].Content);
    points.Add(pt);
}

await qdrantClient.UpsertAsync(collectionName, points);
Console.WriteLine($"[Upserted {points.Count} points into '{collectionName}']\n");

// 5. Query Containing an Exact Policy Code + Semantic Meaning
string userQuery = "What is the deductible for claim POL-4819 regarding water damage?";
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"Target Query: \"{userQuery}\"");
Console.ResetColor();

var queryEmbeddings = await embeddingService.GenerateEmbeddingsAsync([userQuery], kernel: kernel);
var queryVector = queryEmbeddings[0].ToArray();

// A. PURE DENSE VECTOR SEARCH (Notice how generic water damage terms compete)
Console.WriteLine("\n--- 1. Pure Dense Vector Search (Top-2) ---");
#pragma warning disable CS0618
var denseResults = await qdrantClient.SearchAsync(
    collectionName: collectionName,
    vector: queryVector,
    limit: 2
);
#pragma warning restore CS0618

foreach (var hit in denseResults)
{
    Console.WriteLine($"Score: {hit.Score:F4} | Policy: {hit.Payload["policyId"].StringValue} | Content: {hit.Payload["content"].StringValue}");
}

// B. HYBRID SEARCH: Dense Vector + Full-Text Exact Term Match Filter
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("\n--- 2. Hybrid Retrieval (Dense Vector + Full-Text Filter for 'POL-4819') ---");
Console.ResetColor();

var hybridFilter = new Filter();
hybridFilter.Must.Add(new Condition
{
    Field = new FieldCondition
    {
        Key = "content",
        Match = new Match { Text = "pol-4819" } // Exact token match via Full-Text Index
    }
});

#pragma warning disable CS0618
var hybridResults = await qdrantClient.SearchAsync(
    collectionName: collectionName,
    vector: queryVector,
    filter: hybridFilter,
    limit: 1
);
#pragma warning restore CS0618

foreach (var hit in hybridResults)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"Score: {hit.Score:F4} | Policy: {hit.Payload["policyId"].StringValue}");
    Console.WriteLine($"Exact Match Extracted: {hit.Payload["content"].StringValue}");
    Console.ResetColor();
}
Console.WriteLine("----------------------------------------------------------------");
