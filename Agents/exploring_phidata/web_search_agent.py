from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

web_agent = Agent(
    name="Web Agent",
    model=Groq(
        api_key=os.getenv('GROQ_API_KEY'),
        id="llama-3.3-70b-versatile"
    ),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

try:
    web_agent.print_response("What's happening in Bangladesh?", stream=True)
except groq.APIError as e:
    print(f"API Error: {e}")
