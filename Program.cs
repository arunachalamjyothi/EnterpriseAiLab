using System.Diagnostics;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

const string chatModelId = "llama3.2:1b"; // Match your local Ollama chat model

// 1. Setup Semantic Kernel pointing to local Ollama
var httpClient = new HttpClient { BaseAddress = new Uri("http://localhost:11434/v1") };
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId: chatModelId,
    apiKey: "local-ollama",
    httpClient: httpClient
);

Kernel kernel = builder.Build();
var chatService = kernel.GetRequiredService<IChatCompletionService>();

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================");
Console.WriteLine("  Day 11: Two-Stage Retrieval — Cross-Encoder Reranking Spike   ");
Console.WriteLine("================================================================\n");
Console.ResetColor();

// 2. Subtle query where keyword overlap is misleading
string userQuery = "Can I expense a home ergonomic chair if I work hybrid two days a week?";

// 3. Simulated Stage-1 Hybrid Search Top-10 Candidates (Ordered by naive keyword/vector match)
// Notice how noisy keyword-heavy docs ("chair", "expense", "work") rank high initially,
// while the true nuanced answer sits buried at Rank #7.
var stage1Candidates = new List<CandidateChunk>
{
    new("DOC-01", 1, "Office chair maintenance and facility repair requests must be logged on the building helpdesk portal."),
    new("DOC-02", 2, "Expense reports for client dinners require itemized restaurant receipts and attendee names."),
    new("DOC-03", 3, "Hybrid work schedules allow employees to work from home up to two days per week with manager approval."),
    new("DOC-04", 4, "Standing desks and ergonomic chairs in the Chennai and Bengaluru offices are assigned on a hot-desking basis."),
    new("DOC-05", 5, "Travel expense reimbursements are processed every Friday via payroll direct deposit."),
    new("DOC-06", 6, "Employees working from home must ensure their home Wi-Fi meets corporate VPN encryption standards."),
    new("DOC-07", 7, "A one-time $300 home-office hardware stipend is reimbursable for orthopedic seating or monitors, applicable strictly to full-time remote staff—hybrid staff are ineligible."),
    new("DOC-08", 8, "Annual health checkup expenses can be claimed under the corporate wellness wallet."),
    new("DOC-09", 9, "Ergonomic posture training is mandatory for all new hires during onboarding week."),
    new("DOC-10", 10, "IT peripherals such as keyboards and mice are issued directly by the local IT asset desk.")
};

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"User Query: \"{userQuery}\"\n");
Console.ResetColor();

Console.WriteLine("--- Stage 1: Initial Hybrid Search Order (Before Reranking) ---");
foreach (var c in stage1Candidates)
{
    Console.WriteLine($"  Initial Rank #{c.InitialRank:D2} [{c.DocId}]: {c.Text}");
}

// 4. Stage 2: Cross-Encoder Relevance Scoring (Joint Query + Document Evaluation)
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("\n[Running Stage-2 Cross-Encoder Attention Scoring across 10 candidates...]");
Console.ResetColor();

var sw = Stopwatch.StartNew();
var rerankedResults = new List<RerankedChunk>();

// Deterministic settings for strict scoring
var executionSettings = new OpenAIPromptExecutionSettings
{
    Temperature = 0.0,
    MaxTokens = 10
};

foreach (var candidate in stage1Candidates)
{
    // Joint attention prompt evaluating exact semantic entailment between query & chunk
    string crossEncoderPrompt = $"""
        Score the direct relevance of the Document to answering the Query on a scale from 0.00 to 1.00.
        Rules:
        - 0.90 to 1.00: Directly answers the specific condition/eligibility in the query.
        - 0.40 to 0.70: Shares keywords or general topic, but does not answer the specific question.
        - 0.00 to 0.30: Unrelated.
        Output ONLY a single decimal number between 0.00 and 1.00. Do not output any words.

        Query: {userQuery}
        Document: {candidate.Text}
        Score:
        """;

    var response = await chatService.GetChatMessageContentAsync(crossEncoderPrompt, executionSettings);
    string rawScore = response.Content?.Trim() ?? "0.0";

    if (!double.TryParse(rawScore, out double score))
    {
        score = 0.0;
    }

    rerankedResults.Add(new RerankedChunk(
        DocId: candidate.DocId,
        InitialRank: candidate.InitialRank,
        RerankScore: Math.Clamp(score, 0.0, 1.0),
        Text: candidate.Text
    ));
}

sw.Stop();

// 5. Sort by Cross-Encoder Score Descending and Display Top-3
var finalTopK = rerankedResults
    .OrderByDescending(r => r.RerankScore)
    .ToList();

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n================================================================");
Console.WriteLine($"  Stage 2: Reranked Results (Completed in {sw.ElapsedMilliseconds} ms)         ");
Console.WriteLine($"================================================================");
Console.ResetColor();

for (int i = 0; i < finalTopK.Count; i++)
{
    var item = finalTopK[i];
    int newRank = i + 1;
    int rankShift = item.InitialRank - newRank;
    string shiftLabel = rankShift > 0 ? $"+{rankShift} ↑" : (rankShift < 0 ? $"{rankShift} ↓" : "0");

    if (newRank <= 3)
        Console.ForegroundColor = ConsoleColor.Green;
    else
        Console.ForegroundColor = ConsoleColor.DarkGray;

    Console.WriteLine($"Final Rank #{newRank:D2} (Was #{item.InitialRank:D2} | Shift: {shiftLabel,-5}) | Score: {item.RerankScore:F2} | [{item.DocId}]");
    Console.WriteLine($"   {item.Text}\n");
}
Console.ResetColor();

public record CandidateChunk(string DocId, int InitialRank, string Text);
public record RerankedChunk(string DocId, int InitialRank, double RerankScore, string Text);
