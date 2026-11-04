using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

//1. Point to your local Ollama container
string modelId="llama3.2:1b";
var endpoint=new Uri("http://localhost:11434/v1");

var builder=Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId:modelId,
    apiKey:"local-ollama",
    endpoint:endpoint
);

Kernel kernel=builder.Build();
var chatService=kernel.GetRequiredService<IChatCompletionService>();

//2. The Benchmark Prompt
const string prompt="Name 3 unique, innovative names for an AI analytics company. Return only names separated by commas.";

//3. Test Function
async Task RunExperimentAsync(string testName,OpenAIPromptExecutionSettings settings)
{
    Console.ForegroundColor=ConsoleColor.Cyan;
    Console.WriteLine($"\n==============================================================");
Console.WriteLine($"RUNNING: {testName} (Temperature={settings.Temperature},TopP={settings.TopP})");
Console.WriteLine($"==============================================================");
Console.ResetColor();
for(int run=1;run<=3;run++)
{
    var history=new ChatHistory();
    history.AddUserMessage(prompt);

    var response=await chatService.GetChatMessageContentAsync(
        history,
        executionSettings:settings,
        kernel:kernel
    );

    Console.ForegroundColor=ConsoleColor.Green;
    Console.WriteLine($"Run#{run}:{response.Content?.Trim()}");
    Console.ResetColor();
}
}

//4. Experiment A: High Temperature(Creative/Non-deterministic)
var creativeSettings=new OpenAIPromptExecutionSettings
{
    Temperature=1.0,
    MaxTokens=150,
    TopP=1.0
};

await RunExperimentAsync("Experiment A: Creative Drafting", creativeSettings);

//5. Experiment B: Zero Temperature(Deterministic/ Enterprise Extraction)
var deterministicSettings=new OpenAIPromptExecutionSettings
{
    Temperature=0.0,
    MaxTokens=150,
    TopP=0.1
};

await RunExperimentAsync("Experiment B: Strict Determinism",deterministicSettings);


