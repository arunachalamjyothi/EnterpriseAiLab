using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

const string collectionName = "enterprise_policies";
const ulong vectorSize = 768; // nomic-embed-text output dimension

// 1. Initialize Qdrant Client (gRPC port 6334)
var qdrantClient = new QdrantClient("localhost", 6334);

// 2. Configure Semantic Kernel with Local Ollama nomic-embed-text
var httpClient = new HttpClient
{
    BaseAddress = new Uri("http://localhost:11434/v1")
};

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
Console.WriteLine("=== Day 5: Vector Store Ingestion & Retrieval Pipeline ===\n");
Console.ResetColor();

// 3. Ensure Collection Exists (Recreate fresh for the spike)
bool collectionExists = await qdrantClient.CollectionExistsAsync(collectionName);
if (collectionExists)
{
    await qdrantClient.DeleteCollectionAsync(collectionName);
}

await qdrantClient.CreateCollectionAsync(
    collectionName: collectionName,
    vectorsConfig: new VectorParams
    {
        Size = vectorSize,
        Distance = Distance.Cosine
    }
);
Console.WriteLine($"[Collection '{collectionName}' created with vector size {vectorSize} and Cosine distance]");

// 4. Define Mock Enterprise Policies
var policies = new[]
{
    new
    {
        Id = 1UL,
        PolicyId = "POL-HR-001",
        Department = "HR",
        Content = "Medical reimbursement and hospital invoice claims must be submitted via the internal HR portal within 30 days of discharge."
    },
    new
    {
        Id = 2UL,
        PolicyId = "POL-IT-002",
        Department = "IT",
        Content = "Corporate laptops must have full-disk encryption enabled and must not connect to unsecured public Wi-Fi networks."
    },
    new
    {
        Id = 3UL,
        PolicyId = "POL-FIN-003",
        Department = "Finance",
        Content = "Travel allowance claims and domestic flight bookings require line manager pre-approval before corporate card settlement."
    }
};

// 5. Generate Dense Embeddings and Upsert into Qdrant
Console.WriteLine("[Generating embeddings and upserting policy records...]");

var contents = policies.Select(p => p.Content).ToList();
var embeddings = await embeddingService.GenerateEmbeddingsAsync(contents, kernel: kernel);

var points = new List<PointStruct>();
for (int i = 0; i < policies.Length; i++)
{
    var point = new PointStruct
    {
        Id = policies[i].Id,
        Vectors = embeddings[i].ToArray()
    };
    point.Payload.Add("policyId", policies[i].PolicyId);
    point.Payload.Add("department", policies[i].Department);
    point.Payload.Add("text", policies[i].Content);

    points.Add(point);
}

await qdrantClient.UpsertAsync(collectionName, points);
Console.WriteLine($"[Successfully upserted {points.Count} policy records into Qdrant]\n");

// 6. Semantic Query: Fetch Top-1 Nearest Neighbor
string query = "How do I submit hospital invoices?";
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"User Query: \"{query}\"");
Console.ResetColor();

var queryEmbeddings = await embeddingService.GenerateEmbeddingsAsync([query], kernel: kernel);
var queryVector = queryEmbeddings[0].ToArray();

var searchResults = await qdrantClient.SearchAsync(
    collectionName: collectionName,
    vector: queryVector,
    limit: 1
);

// 7. Display Top Result and Score
Console.WriteLine("--------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Green;
foreach (var match in searchResults)
{
    Console.WriteLine($"Top Match Score : {match.Score:F4} (Cosine Similarity)");
    Console.WriteLine($"Policy ID       : {match.Payload["policyId"].StringValue}");
    Console.WriteLine($"Department      : {match.Payload["department"].StringValue}");
    Console.WriteLine($"Matched Policy  : {match.Payload["text"].StringValue}");
}
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------");
