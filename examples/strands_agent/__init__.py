"""SemStash Agent - AI agent for semantic storage using Strands SDK.

Example usage:
    from examples.strands_agent import semstash_agent

    with semstash_agent(bucket="my-bucket") as agent:
        agent("Upload photo.jpg and search for similar images")
"""

from examples.strands_agent.agent import semstash_agent

__all__ = ["semstash_agent"]
