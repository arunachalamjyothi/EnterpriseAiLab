using System.Text.Json;
using Microsoft.SemanticKernel.Text;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================");
Console.WriteLine("  Day 8: Recursive Sliding Window Chunker with Metadata Links   ");
Console.WriteLine("================================================================\n");
Console.ResetColor();

// 1. Unstructured enterprise source policy
string rawPolicyDocument = 
    "SECTION 4: CORPORATE HEALTH INSURANCE & HOSPITALIZATION COVERAGE. " +
    "Eligible employees and their registered dependents are covered under the Group Medical Floater Policy. " +
    "For planned hospitalizations, pre-authorization must be obtained at least 72 hours prior to hospital admission through the TPA desk. " +
    "In emergency admissions, intimation must reach the insurer within 24 hours of hospitalization. " +
    "Original hospital discharge summaries, pharmacy invoices, and diagnostic reports must be submitted within 30 days of discharge. " +
    "Claims submitted beyond 30 calendar days shall be rejected automatically unless accompanied by a written exception waiver approved by the HR Director. " +
    "Room rent coverage is capped at 1% of the total sum insured per day for standard rooms and 2% per day for intensive care units (ICU). " +
    "Non-medical expenses including registration fees, PPE kits, and administrative charges are non-reimbursable and must be settled directly with the hospital.";

const string documentId = "POL-HR-HEALTH-2026";
const int pageIndex = 4;
const int maxTokensPerChunk = 45; // Compact limit to demonstrate multi-chunk splitting
const int overlapTokens = 15;     // Sliding window overlap boundary

#pragma warning disable SKEXP0050
// 2. Step 1: Split into lines/sentences respecting boundaries
var lines = TextChunker.SplitPlainTextLines(rawPolicyDocument, maxTokensPerLine: 25);

// 3. Step 2: Combine lines into paragraphs with sliding overlap
var rawChunks = TextChunker.SplitPlainTextParagraphs(
    lines, 
    maxTokensPerParagraph: maxTokensPerChunk, 
    overlapTokens: overlapTokens
);
#pragma warning restore SKEXP0050

// 4. Step 3: Wrap chunks with enterprise metadata pointers
var taggedChunks = new List<PolicyChunkMetadata>();
string? previousChunkId = null;

for (int i = 0; i < rawChunks.Count; i++)
{
    string chunkId = $"{documentId}-P{pageIndex}-C{i + 1:D2}";
    
    var chunkRecord = new PolicyChunkMetadata(
        ChunkId: chunkId,
        DocumentId: documentId,
        PageIndex: pageIndex,
        ChunkIndex: i + 1,
        PreviousChunkId: previousChunkId,
        TokenEstimate: EstimateTokenCount(rawChunks[i]),
        Text: rawChunks[i]
    );

    taggedChunks.Add(chunkRecord);
    previousChunkId = chunkId;
}

// 5. Display Chunk Output with Context Links
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"[Successfully Split Document into {taggedChunks.Count} Overlapping Chunks]\n");
Console.ResetColor();

foreach (var chunk in taggedChunks)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"--- Chunk ID: {chunk.ChunkId} (Tokens: ~{chunk.TokenEstimate}) ---");
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"  DocId: {chunk.DocumentId} | Page: {chunk.PageIndex} | PrevChunkId: {chunk.PreviousChunkId ?? "NONE (Root)"}");
    Console.ResetColor();
    Console.WriteLine($"  Text: \"{chunk.Text}\"\n");
}

// Quick token counter approximation (1 token ≈ 4 characters)
static int EstimateTokenCount(string text) => (int)Math.Ceiling(text.Length / 4.0);

// Record definition for Qdrant payload serialization
public record PolicyChunkMetadata(
    string ChunkId,
    string DocumentId,
    int PageIndex,
    int ChunkIndex,
    string? PreviousChunkId,
    int TokenEstimate,
    string Text
);
