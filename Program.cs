using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

// 1. Configure HttpClient pointing directly to Ollama's OpenAI-compatible endpoint
var httpClient=new HttpClient
{
    BaseAddress=new Uri("http://localhost:11434/v1")
};

string modelId="nomic-embed-text";

//2. Register embedding generation using the standard HttpClient overload
var builder=Kernel.CreateBuilder();
builder.AddOpenAITextEmbeddingGeneration(
    modelId:modelId,
    apiKey: "local-ollama",
    httpClient:httpClient
);

Kernel kernel=builder.Build();

//3. Resolve the Embedding Service
var embeddingService=kernel.GetRequiredService<ITextEmbeddingGenerationService>();

//4. Target Phrases
string phrase1="Client claims medical reimbursement";
string phrase2="Health insurance payout request";
string phrase3="The quick brown fox jumps over the lazy dog"; //Negative baseline

Console.ForegroundColor=ConsoleColor.Cyan;
Console.WriteLine("[Generating Dense Embeddings via local nomic-embed-text...]\n");
Console.ResetColor();

//5. Generate Vectors
var embeddings=await embeddingService.GenerateEmbeddingsAsync(
    [phrase1,phrase2,phrase3],
    kernel:kernel
);

ReadOnlyMemory<float> vector1=embeddings[0];
ReadOnlyMemory<float> vector2=embeddings[1];
ReadOnlyMemory<float> vector3=embeddings[2];

Console.WriteLine($"Vector 1 Dimension:{vector1.Length}floats");
Console.WriteLine($"Vector 2 Dimension:{vector2.Length}floats");
Console.WriteLine("-----------------------------------------");

//6. Calculate Cosine Similatities
float simRelated=CalculateCosineSimilarity(vector1.Span, vector2.Span);
float simUnrelated=CalculateCosineSimilarity(vector1.Span, vector3.Span);

//7. Output Results & Verification
Console.ForegroundColor=ConsoleColor.Green;
Console.WriteLine($"Phrase 1:\"{phrase1}\"");
Console.WriteLine($"Phrase 2:\"{phrase2}\"");
Console.WriteLine($"Cosine Similarity: {simRelated:F4} (Expected:>0.80 - High Semantic Alignment)");
Console.ResetColor();

Console.WriteLine("\n-------------------------------------------------");
Console.ForegroundColor=ConsoleColor.Yellow;
Console.WriteLine($"Phrase 1:\"{phrase1}\"");
Console.WriteLine($"Phrase 3:\"{phrase3}\"");
Console.WriteLine($"Cosine Similarity: {simUnrelated:F4}(Expected: Low Semantic Alignment)");
Console.ResetColor();

// Helper method: Cosine Similarity calculation in C#
static float CalculateCosineSimilarity(ReadOnlySpan<float> vectorA, ReadOnlySpan<float> vectorB)
{
    if(vectorA.Length!=vectorB.Length)
    {
        throw new ArgumentException("Vectors must have the same dimension length.");

    }

    float dotProduct=0.0f;
    float normA=0.0f;
    float normB=0.0f;

    for(int i=0;i<vectorA.Length;i++)
    {
        dotProduct+=vectorA[i]*vectorB[i];
        normA+=vectorA[i]*vectorA[i];
        normB+=vectorB[i]*vectorB[i];
    }

    if(normA<=0.0f||normB<=0.0f)
    {
        return 0.0f;
    }

    return dotProduct/((float)Math.Sqrt(normA)*(float)Math.Sqrt(normB));
}