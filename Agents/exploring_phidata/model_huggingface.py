
from phi.agent import Agent, RunResponse
from phi.model.huggingface import HuggingFaceChat

agent = Agent(
    model=HuggingFaceChat(
        id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens=4096,
    ),
    markdown=True
)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story.")
# print(run.content)

# Print the response on the terminal
agent.print_response("Explain me linear regression in simple math")

