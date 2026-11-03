using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

// 1. Point to your local Ollama instance
// Ollama provides an OpenAI-compatible endpoint at /v1
string modelId = "llama3.2:1b";
var endpoint = new Uri("http://localhost:11434/v1");

// 2. Build the Semantic Kernel
// apiKey can be any non-empty placeholder string (Ollama does not enforce auth)
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId: modelId,
    apiKey: "local-ollama",
    endpoint: endpoint
);

Kernel kernel = builder.Build();

// 3. Resolve the Chat Completion Service
var chatService = kernel.GetRequiredService<IChatCompletionService>();

// 4. Construct System Instructions & User Prompt
var history = new ChatHistory();
history.AddSystemMessage("""
    You are an Enterprise AI Architect assistant.
    Provide concise, production-ready technical responses without conversational filler.
    """);

history.AddUserMessage("Explain why output tokens cost more than input tokens in 2 bullet points.");

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine($"[Invoking local {modelId} via Semantic Kernel & Ollama on Docker...]\n");
Console.ResetColor();

// 5. Invoke the local model
var response = await chatService.GetChatMessageContentAsync(
    chatHistory: history,
    kernel: kernel
);

// 6. Display the Output
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine(response.Content);
Console.ResetColor();

Console.WriteLine("\n-------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[Success: Model executed locally with 0 cloud API costs]");
Console.ResetColor();