"""SemStash Agent - AI agent for semantic storage using Strands SDK.

NOTE: The SemStashAgent class is now part of the main semstash package.
This module re-exports from semstash.agent for backwards compatibility.

Example usage:
    # Preferred (from main package)
    from semstash.agent import SemStashAgent, semstash_agent

    # Or via this example module (backwards compatible)
    from examples.strands_agent import semstash_agent

    with semstash_agent(bucket="my-bucket") as agent:
        response = agent.chat("What files are stored?")
        print(response)
"""

from semstash.agent import SemStashAgent, semstash_agent

__all__ = ["SemStashAgent", "semstash_agent"]
