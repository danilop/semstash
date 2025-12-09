"""Example of using SemStash Agent for semantic storage.

This example demonstrates how to use the SemStash Agent API to interact
with content stored in a SemStash bucket via conversational AI.

The SemStashAgent class is now part of the main semstash package.
"""

# Re-export from the main package for backwards compatibility
from semstash.agent import SemStashAgent, semstash_agent

__all__ = ["SemStashAgent", "semstash_agent"]


def main() -> None:
    """Demonstrate SemStash Agent usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <bucket-name>")
        print("\nExample:")
        print("  python agent.py my-stash")
        sys.exit(1)

    bucket_name = sys.argv[1]

    print(f"Connecting to SemStash bucket: {bucket_name}")
    print("=" * 50)

    with semstash_agent(bucket_name) as agent:
        print("\nAgent ready! Try these commands:")
        print("  - 'browse /'")
        print("  - 'search for documents about...'")
        print("  - 'get stats'")
        print("  - 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "/quit"):
                    print("Goodbye!")
                    break

                # Use non-streaming for simplicity in this example
                response = agent.chat(user_input)
                print(f"\nAgent: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
