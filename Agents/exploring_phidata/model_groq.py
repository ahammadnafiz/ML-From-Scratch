from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the agent with API key from environment
agent = Agent(
    model=Groq(
        api_key=os.getenv('GROQ_API_KEY'),
        id="llama-3.3-70b-versatile"
    ),
    markdown=True
)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story.")
# print(run.content)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")
