from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent
from typing import Optional, Type, Any, List
import math
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False
from pydantic import BaseModel


class LLMAgent:
    def __init__(
        self,
        llm_api_key: str,
        llm_deployment_name: str,
        llm_endpoint: str,
        agent_name: str,
        system_prompt: str,
        plugin: Optional[Any] = None,
        response_format: Optional[Type[BaseModel]] = None,
        memory_max_tokens: int = 10_000,
        **kwargs,
    ):
        """Initialize the LLMAgent.

        Args:
            llm_api_key: The API key for the LLM service.
            llm_deployment_name: The deployment name for the LLM.
            llm_endpoint: The endpoint URL for the LLM service.
            agent_name: The name of the agent.
            system_prompt: The system prompt for the agent.
            plugins: the class defining the agent's plugins.
            response_format: the class defining the agent's response format.
            **kwargs: Additional keyword arguments for ChatCompletionAgent.
        """
        azure_chat_completion = AzureChatCompletion(
            api_key=llm_api_key,
            deployment_name=llm_deployment_name,
            endpoint=llm_endpoint,
        )
        kernel = Kernel()
        kernel.add_service(azure_chat_completion)
        execution_settings = AzureChatPromptExecutionSettings()
        if plugin:
            kernel.add_plugin(plugin)
            execution_settings.tool_choice = "auto"
            execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke=True, filters={})
        if response_format:
            execution_settings.response_format = response_format
        
        self.llm_agent = ChatCompletionAgent(
            name=agent_name,
            description=llm_deployment_name + " agent",
            instructions=system_prompt,
            kernel=kernel,
            arguments=KernelArguments(settings=execution_settings),
            **kwargs,
        )

        # --- Conversation Memory via ChatHistoryAgentThread ---
        self._memory_max_tokens = int(memory_max_tokens)
        self._encoding = None
        if _HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None
        # Thread maintains ChatHistory internally
        self._thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        # Independent lightweight history we control (list[{"role","content"}])
        # because ChatHistoryAgentThread.chat_history may remain None until after
        # certain internal events, causing /memstats to show zeros.
        self._history: list[dict] = []

    # ------------- Memory Helpers using thread history -------------
    def reset_memory(self):
        """Reset the chat history thread."""
        self._thread = ChatHistoryAgentThread()
        self._history.clear()

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        return max(1, math.ceil(len(text) / 4))

    def _current_token_count(self) -> int:
        """Estimate total tokens across our maintained history list."""
        total = 0
        for entry in self._history:
            total += self._estimate_tokens(entry.get("content", ""))
        return total

    def _prune_if_needed(self):
        if self._memory_max_tokens <= 0:
            return
        # Remove oldest non-system messages from our list until within budget
        while self._current_token_count() > self._memory_max_tokens and len(self._history) > 1:
            removed = False
            for i, msg in enumerate(self._history):
                if msg.get("role") != "system":
                    del self._history[i]
                    removed = True
                    break
            if not removed:
                break

    # -------- Public Introspection Helpers --------
    def get_history(self, limit: int | None = None) -> List[dict]:
        """Return list of message dicts from maintained history."""
        if limit is not None and limit > 0:
            return self._history[-limit:]
        return list(self._history)

    def memory_stats(self) -> dict:
        """Return statistics about current memory usage."""
        total_tokens = self._current_token_count()
        messages = len(self.get_history())
        return {
            "messages": messages,
            "token_estimate": total_tokens,
            "token_limit": self._memory_max_tokens,
            "utilization_pct": round(100 * total_tokens / self._memory_max_tokens, 2) if self._memory_max_tokens else None,
        }

    async def __call__(self, user_prompt: str, **kwargs):
        """Call the agent with a user prompt, including conversation memory.

        Memory retention:
          - Previous user/assistant messages are replayed before the new user prompt.
          - After receiving assistant response, both new user and assistant messages
            are appended and the memory is trimmed to the configured token budget.
        """
        # Add new user message; ChatHistoryAgentThread will accumulate internally
        new_message = ChatMessageContent(role="user", content=user_prompt)
        # Track user message in our own history immediately
        self._history.append({"role": "user", "content": user_prompt})
        intermediate_steps: list[ChatMessageContent] = []

        async def handle_intermediate_steps(message: ChatMessageContent) -> None:
            intermediate_steps.append(message)

        thread = None
        agent_response = {}

        # Use persistent thread; supply existing thread each call
        if thread is None:
            thread = self._thread
        async for response in self.llm_agent.invoke(
            messages=new_message,
            thread=thread,
            on_intermediate_message=handle_intermediate_steps,
        ):
            agent_response["name"] = response.name
            agent_response["content"] = response.content.inner_content.choices[0].message.content
            # Update stored thread reference in case underlying implementation returns a new one
            if hasattr(response, "thread") and response.thread is not None:
                self._thread = response.thread  # type: ignore[assignment]

        # Track assistant message content in our own history
        if agent_response.get("content"):
            self._history.append({"role": "assistant", "content": agent_response["content"]})
        # Prune after adding assistant response
        self._prune_if_needed()

        # Collect all function/tool call attempts in order with their arguments and eventual results
        intermediate_messages = {"function_calls": []}
        for msg in intermediate_steps:
            if any(isinstance(item, FunctionResultContent) for item in msg.items):
                for fr in msg.items:
                    if isinstance(fr, FunctionResultContent):
                        for entry in reversed(intermediate_messages["function_calls"]):
                            if entry["name"] == fr.name and entry["result"] is None:
                                entry["result"] = fr.result
                                break
                        else:
                            intermediate_messages["function_calls"].append({
                                "name": fr.name,
                                "arguments": None,
                                "result": fr.result,
                            })
            elif any(isinstance(item, FunctionCallContent) for item in msg.items):
                for fcc in msg.items:
                    if isinstance(fcc, FunctionCallContent):
                        intermediate_messages["function_calls"].append({
                            "name": fcc.name,
                            "arguments": fcc.arguments,
                            "result": None,
                        })
            else:
                pass
        return agent_response, intermediate_messages
