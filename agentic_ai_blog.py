import os
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List

import openai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODEL = os.getenv("AGENTIC_AI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_CODE_MODEL = os.getenv("AGENTIC_AI_CODE_MODEL", "gpt-4o-mini")
DEFAULT_WRITER_MODEL = os.getenv("AGENTIC_AI_WRITER_MODEL", "gpt-4o-mini")
DEFAULT_OPENROUTER_BASE = os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"


class OpenAIClient:
    """A compatibility wrapper for OpenAI and OpenRouter Python SDKs."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env or environment."
            )

        self.api_base = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENROUTER_API_BASE")
            or DEFAULT_OPENROUTER_BASE
        )
        self.default_model = os.getenv("AGENTIC_AI_MODEL", DEFAULT_CHAT_MODEL)
        self.client = None

        openai.api_key = self.api_key
        openai.api_base = self.api_base

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> str:
        model = model or self.default_model

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response:
            raise RuntimeError("Empty response from the LLM provider.")

        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content.strip()

        if hasattr(response, "output") and response.output:
            first_item = response.output[0]
            if "content" in first_item:
                content = first_item["content"]
                if isinstance(content, list) and content:
                    return content[0].get("text", "").strip()
                return str(content).strip()

        raise RuntimeError("Unable to parse response from the LLM provider.")


@dataclass
class ConditionalState:
    user_request: str
    route: Optional[str] = None
    final_response: Optional[str] = None


class ConditionalAgenticWorkflow:
    """A conditional routing workflow that selects one specialist agent."""

    ROUTES = {"chat", "code", "writer"}

    def __init__(self, client: OpenAIClient):
        self.client = client
        self.chat_model = os.getenv("AGENTIC_AI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.code_model = os.getenv("AGENTIC_AI_CODE_MODEL", DEFAULT_CODE_MODEL)
        self.writer_model = os.getenv("AGENTIC_AI_WRITER_MODEL", DEFAULT_WRITER_MODEL)

    def route_request(self, user_request: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a request routing agent. Classify the user request into one of three routes."
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{user_request}\n\n"
                    "ROUTES:\n"
                    "- chat: simple conversation, greetings, explanations, or general questions\n"
                    "- code: programming, debugging, writing code, or software development tasks\n"
                    "- writer: current events, online research, web content, articles, or writing-based requests\n\n"
                    "STRICT OUTPUT RULE:\n"
                    "Return only one word: chat, code, or writer."
                ),
            },
        ]

        response_text = self.client.create_chat_completion(
            prompt,
            model=self.chat_model,
            temperature=0.0,
            max_tokens=30,
        )

        route = response_text.strip().lower()
        if route not in self.ROUTES:
            route = "chat"
        return route

    def simple_chat_agent(self, user_request: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful and friendly chat assistant."
            },
            {
                "role": "user",
                "content": f"USER REQUEST:\n{user_request}\n\nTASK:\nAnswer clearly and naturally."
            },
        ]
        return self.client.create_chat_completion(
            prompt,
            model=self.chat_model,
            temperature=0.7,
            max_tokens=800,
        )

    def code_agent(self, user_request: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a senior software engineering assistant."
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{user_request}\n\n"
                    "TASK:\nProvide a practical coding answer. Include code examples when useful."
                ),
            },
        ]
        return self.client.create_chat_completion(
            prompt,
            model=self.code_model,
            temperature=0.2,
            max_tokens=1200,
        )

    def writer_agent(self, user_request: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a professional research and content writing assistant."
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{user_request}\n\n"
                    "TASK:\nWrite a clear, well-structured response for the user's writing request. "
                    "If live browsing is unavailable, say so and answer based on the request alone."
                ),
            },
        ]
        return self.client.create_chat_completion(
            prompt,
            model=self.writer_model,
            temperature=0.7,
            max_tokens=1200,
        )

    def generate_response(self, user_request: str) -> ConditionalState:
        if not user_request or not user_request.strip():
            raise ValueError("The prompt cannot be empty.")

        state = ConditionalState(user_request=user_request.strip())
        state.route = self.route_request(state.user_request)

        if state.route == "code":
            state.final_response = self.code_agent(state.user_request)
        elif state.route == "writer":
            state.final_response = self.writer_agent(state.user_request)
        else:
            state.final_response = self.simple_chat_agent(state.user_request)

        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic AI conditional workflow from the command line.")
    parser.add_argument("--prompt", required=True, help="The request or topic to process.")
    parser.add_argument("--show-route", action="store_true", help="Show the selected agent route.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAIClient()
    workflow = ConditionalAgenticWorkflow(client)
    state = workflow.generate_response(args.prompt)

    if args.show_route:
        print(f"Selected route: {state.route}\n")

    print("=== Final Response ===\n")
    print(state.final_response)


if __name__ == "__main__":
    main()
