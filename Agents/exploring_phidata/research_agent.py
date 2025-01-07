from textwrap import dedent
from datetime import datetime
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.exa import ExaTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

agent = Agent(
    model=Groq(
        api_key=os.getenv('GROQ_API_KEY'),
        id="llama-3.3-70b-versatile"  # Updated model ID
    ),
    tools=[ExaTools(start_published_date=datetime.now().strftime("%Y-%m-%d"), type="keyword")],
    description="You are an advanced AI researcher writing a report on a topic.",
    instructions=[
        "For the provided topic, run 3 different searches.",
        "Read the results carefully and prepare a NYT worthy report.",
        "Focus on facts and make sure to provide references.",
    ],
    expected_output=dedent("""\
    An engaging, informative, and well-structured report in markdown format:
    ## Engaging Report Title
    ### Overview
    {give a brief introduction of the report and why the user should read this report}
    {make this section engaging and create a hook for the reader}
    ### Section 1
    {break the report into sections}
    {provide details/facts/processes in this section}
    ... more sections as necessary...
    ### Takeaways
    {provide key takeaways from the article}
    ### References
    - [Reference 1](link)
    - [Reference 2](link)
    - [Reference 3](link)
    - published on {date} in dd/mm/yyyy
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    save_response_to_file="tmp/{message}.md",
)

# Run the agent
agent.print_response("Simulation theory", stream=True)
