
import os
from dotenv import load_dotenv
from openai import OpenAI, APIError

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-5.5")

if not api_key:
    raise ValueError("Please configure OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are Jarvis, a personal AI assistant.
Be calm, intelligent, precise, and helpful.
Give clear answers and explain technical topics simply.
Remember the conversation context provided to you.
Never claim to perform actions you have not performed.
"""

conversation = []


def chat(message):
    conversation.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=conversation[-20:],
            max_output_tokens=1000
        )

        answer = response.output_text.strip()

        if not answer:
            answer = "I couldn't generate a response."

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        # Keep the most recent 20 messages
        if len(conversation) > 20:
            del conversation[:-20]

        return answer

    except APIError as error:
        conversation.pop()
        return f"OpenAI API error: {error}"


def main():
    print("=" * 40)
    print("       JARVIS AI ASSISTANT")
    print("=" * 40)
    print("Type 'exit' to quit.")
    print("Type 'clear' to reset conversation.\n")

    while True:
        try:
            message = input("You: ").strip()

            if not message:
                continue

            if message.lower() in ("exit", "quit"):
                print("Jarvis: Goodbye!")
                break

            if message.lower() == "clear":
                conversation.clear()
                print("Jarvis: Conversation cleared.\n")
                continue

            answer = chat(message)
            print(f"\nJarvis: {answer}\n")

        except KeyboardInterrupt:
            print("\nJarvis: Shutting down.")
            break


if __name__ == "__main__":
    main()
