from collections.abc import Callable, Iterable
from typing import Any, Dict, List, MutableMapping, Optional, Type, cast

import math

from pydantic import BaseModel

from agent_framework import ToolProtocol, ChatAgent
from agent_framework import (
    FunctionCallContent as AFFunctionCallContent,
    FunctionResultContent as AFFunctionResultContent,
)
from agent_framework.azure import AzureOpenAIChatClient, AzureAIAgentClient

from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import (
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    AuthorRole
)
from semantic_kernel.functions import KernelArguments

try:
    import tiktoken  # type: ignore

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TIKTOKEN = False


ToolLike = ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]


class SKAgent:
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
        **kwargs: Any,
    ) -> None:
        """Initialize the Semantic Kernel chat agent."""

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
            execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
                auto_invoke=True,
                filters={},
            )
        if response_format:
            execution_settings.response_format = response_format

        self.llm_agent = ChatCompletionAgent(
            name=agent_name,
            description=f"{llm_deployment_name} agent",
            instructions=system_prompt,
            kernel=kernel,
            arguments=KernelArguments(settings=execution_settings),
            **kwargs,
        )

        self._memory_max_tokens = int(memory_max_tokens)
        self._encoding = None
        if _HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

        self._thread: ChatHistoryAgentThread = ChatHistoryAgentThread()
        self._history: list[dict[str, Any]] = []

    def reset_memory(self) -> None:
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
        total = 0
        for entry in self._history:
            total += self._estimate_tokens(entry.get("content", ""))
        return total

    def _prune_if_needed(self) -> None:
        if self._memory_max_tokens <= 0:
            return
        while self._current_token_count() > self._memory_max_tokens and len(self._history) > 1:
            removed = False
            for idx, msg in enumerate(self._history):
                if msg.get("role") != "system":
                    del self._history[idx]
                    removed = True
                    break
            if not removed:
                break

    def get_history(self, limit: int | None = None) -> List[dict[str, Any]]:
        if limit is not None and limit > 0:
            return self._history[-limit:]
        return list(self._history)

    def memory_stats(self) -> dict[str, Any]:
        total_tokens = self._current_token_count()
        messages = len(self.get_history())
        return {
            "messages": messages,
            "token_estimate": total_tokens,
            "token_limit": self._memory_max_tokens,
            "utilization_pct": round(100 * total_tokens / self._memory_max_tokens, 2)
            if self._memory_max_tokens
            else None,
        }

    async def __call__(self, user_prompt: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        new_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)
        self._history.append({"role": "user", "content": user_prompt})
        intermediate_steps: list[ChatMessageContent] = []

        async def handle_intermediate_steps(message: ChatMessageContent) -> None:
            intermediate_steps.append(message)

        thread: ChatHistoryAgentThread | None = self._thread
        agent_response: dict[str, Any] = {}

        async for response in self.llm_agent.invoke(
            messages=new_message,
            thread=thread,
            on_intermediate_message=handle_intermediate_steps,
        ):
            agent_response["name"] = response.name
            if response.content.inner_content and response.content.inner_content.choices:
                agent_response["content"] = response.content.inner_content.choices[0].message.content
            else:
                agent_response["content"] = ""
            if hasattr(response, "thread") and response.thread is not None:
                self._thread = response.thread  # type: ignore[assignment]

        if agent_response.get("content"):
            self._history.append({"role": "assistant", "content": agent_response["content"]})
        self._prune_if_needed()

        intermediate_messages: dict[str, list[dict[str, Any]]] = {"function_calls": []}
        for msg in intermediate_steps:
            if any(isinstance(item, FunctionResultContent) for item in msg.items):
                for fr in msg.items:
                    if isinstance(fr, FunctionResultContent):
                        for entry in reversed(intermediate_messages["function_calls"]):
                            if entry["name"] == fr.name and entry["result"] is None:
                                entry["result"] = fr.result
                                break
                        else:
                            intermediate_messages["function_calls"].append(
                                {"name": fr.name, "arguments": None, "result": fr.result}
                            )
            elif any(isinstance(item, FunctionCallContent) for item in msg.items):
                for fcc in msg.items:
                    if isinstance(fcc, FunctionCallContent):
                        intermediate_messages["function_calls"].append(
                            {"name": fcc.name, "arguments": fcc.arguments, "result": None}
                        )

        return agent_response, intermediate_messages


class AFAgent:
    def __init__(
        self,
        llm_api_key: str,
        llm_deployment_name: str,
        llm_endpoint: str,
        agent_name: str,
        system_prompt: str,
        tools: Optional[Iterable[ToolLike]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        memory_max_tokens: int = 10_000,
        **kwargs: Any,
    ) -> None:
        """Initialize an agent backed by the Microsoft Agent Framework SDK."""

        self._client = AzureOpenAIChatClient(
            api_key=llm_api_key,
            deployment_name=llm_deployment_name,
            endpoint=llm_endpoint,
        )

        tools_list = list(tools) if tools else []
        if not tools_list:
            tools_argument: ToolLike | list[ToolLike] | None = None
        elif len(tools_list) == 1:
            tools_argument = cast(ToolLike, tools_list[0])
        else:
            tools_argument = cast(list[ToolLike], tools_list)

        agent_kwargs = dict(kwargs)

        self._run_kwargs: dict[str, Any] = {}
        if response_format is not None:
            self._run_kwargs["response_format"] = response_format

        tool_choice_default = agent_kwargs.pop("tool_choice", None)
        if tool_choice_default is None and tools_list:
            tool_choice_default = "auto"
        if tool_choice_default is not None:
            self._run_kwargs["tool_choice"] = tool_choice_default

        self.llm_agent = self._client.create_agent(
            name=agent_name,
            instructions=system_prompt,
            tools=tools_argument,
            **agent_kwargs,
        )

        self._memory_max_tokens = int(memory_max_tokens)
        self._encoding = None
        if _HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

        self._thread = self.llm_agent.get_new_thread()
        self._history = []

    def reset_memory(self) -> None:
        """Reset the Agent Framework conversation thread and clear local history."""

        self._thread = self.llm_agent.get_new_thread()
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
        total = 0
        for entry in self._history:
            total += self._estimate_tokens(entry.get("content", ""))
        return total

    def _prune_if_needed(self) -> None:
        if self._memory_max_tokens <= 0:
            return
        while self._current_token_count() > self._memory_max_tokens and len(self._history) > 1:
            removed = False
            for idx, msg in enumerate(self._history):
                if msg.get("role") != "system":
                    del self._history[idx]
                    removed = True
                    break
            if not removed:
                break

    def get_history(self, limit: int | None = None) -> List[dict[str, Any]]:
        if limit is not None and limit > 0:
            return self._history[-limit:]
        return list(self._history)

    def memory_stats(self) -> dict[str, Any]:
        total_tokens = self._current_token_count()
        messages = len(self.get_history())
        return {
            "messages": messages,
            "token_estimate": total_tokens,
            "token_limit": self._memory_max_tokens,
            "utilization_pct": round(100 * total_tokens / self._memory_max_tokens, 2)
            if self._memory_max_tokens
            else None,
        }

    async def __call__(self, user_prompt: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self._history.append({"role": "user", "content": user_prompt})

        merged_kwargs = {**self._run_kwargs, **kwargs}
        response = await self.llm_agent.run(
            messages=user_prompt,
            thread=self._thread,
            **merged_kwargs,
        )

        response_text = getattr(response, "text", None)
        if response_text is None and hasattr(response, "output"):
            response_text = getattr(getattr(response, "output"), "text", None)

        agent_response: dict[str, Any] = {
            "name": getattr(self.llm_agent, "name", "agent"),
            "content": response_text or "",
        }

        if agent_response["content"]:
            self._history.append({"role": "assistant", "content": agent_response["content"]})
            self._prune_if_needed()

        intermediate_messages: Dict[str, List[dict[str, Any]]] = {"function_calls": []}
        call_index: Dict[str, dict[str, Any]] = {}

        for message in getattr(response, "messages", []) or []:
            for content in message.contents:
                if isinstance(content, AFFunctionCallContent):
                    entry = {
                        "name": content.name,
                        "arguments": content.parse_arguments(),
                        "result": None,
                    }
                    if getattr(content, "call_id", None):
                        call_index[content.call_id] = entry
                    intermediate_messages["function_calls"].append(entry)
                elif isinstance(content, AFFunctionResultContent):
                    target = call_index.get(content.call_id)
                    if target is not None:
                        target["result"] = content.result
                    else:
                        intermediate_messages["function_calls"].append(
                            {"name": None, "arguments": None, "result": content.result}
                        )

        return agent_response, intermediate_messages

class AFAFAgent:
    """Agent Framework wrapper backed by a persistent Azure AI Foundry agent.

    This agent mirrors the behavior of :class:`AFAgent`, but instead of using
    the Azure OpenAI Assistants service it targets Azure AI Foundry's Agents
    endpoint. Use :meth:`AFAFAgent.create` to provision or reuse a persistent
    agent before wiring it into the Microsoft Agent Framework runtime.
    """

    _credential: AsyncAzureCliCredential
    _owns_credential: bool
    _client: AzureAIAgentClient
    llm_agent: ChatAgent
    agent_id: str
    _project_endpoint: str
    _model: str
    _system_prompt: str
    _memory_max_tokens: int
    _run_kwargs: dict[str, Any]
    _history: list[dict[str, Any]]
    _llm_api_key: str
    _encoding: Any | None
    _thread: Any

    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - guard rail
        raise RuntimeError("Use AFAFAgent.create(...) to instantiate this class.")

    @classmethod
    async def create(
        cls,
        llm_api_key: str,
        llm_deployment_name: str,
        llm_endpoint: str,
        agent_name: str,
        system_prompt: str,
        tools: Optional[Iterable[ToolLike]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        memory_max_tokens: int = 10_000,
        recreate_agent: bool = False,
        **kwargs: Any,
    ) -> "AFAFAgent":
        project_endpoint = kwargs.pop("project_endpoint", llm_endpoint)
        if not project_endpoint:
            raise ValueError("Azure AI Foundry project endpoint is required.")

        credential: AsyncAzureCliCredential | None = kwargs.pop("credential", None)
        if credential is None:
            credential = kwargs.pop("async_credential", None)
        explicit_agent_id: str | None = kwargs.pop("agent_id", None)

        owns_credential = False
        if credential is None:
            credential = AsyncAzureCliCredential()
            owns_credential = True

        try:
            async with AsyncAIProjectClient(endpoint=project_endpoint, credential=credential) as project_client:
                agents_client = getattr(project_client, "agents", None)
                if agents_client is None:
                    raise RuntimeError("The Azure AI Projects client does not expose an agents endpoint.")

                if explicit_agent_id:
                    if recreate_agent:
                        try:
                            await agents_client.delete_agent(explicit_agent_id)
                        except ResourceNotFoundError:
                            pass
                        created_agent = await agents_client.create_agent(
                            model=llm_deployment_name,
                            name=agent_name,
                            instructions=system_prompt,
                        )
                        persistent_agent_id = str(created_agent.id)
                    else:
                        existing_agent = await agents_client.get_agent(explicit_agent_id)
                        persistent_agent_id = str(existing_agent.id)
                        if system_prompt and getattr(existing_agent, "instructions", None) != system_prompt:
                            await agents_client.update_agent(persistent_agent_id, instructions=system_prompt)
                else:
                    existing_agent = await cls._find_agent_by_name(agents_client, agent_name)
                    if existing_agent is not None and recreate_agent:
                        existing_id = getattr(existing_agent, "id", None)
                        if existing_id is not None:
                            try:
                                await agents_client.delete_agent(existing_id)
                            except ResourceNotFoundError:
                                pass
                        existing_agent = None

                    if existing_agent is None:
                        created_agent = await agents_client.create_agent(
                            model=llm_deployment_name,
                            name=agent_name,
                            instructions=system_prompt,
                        )
                        persistent_agent_id = str(created_agent.id)
                    else:
                        persistent_agent_id = str(existing_agent.id)
                        if system_prompt and getattr(existing_agent, "instructions", None) != system_prompt:
                            await agents_client.update_agent(persistent_agent_id, instructions=system_prompt)
        except ResourceNotFoundError as exc:  # pragma: no cover - network failure guard
            if owns_credential:
                await credential.close()
            raise RuntimeError(
                "Azure AI Foundry project endpoint returned 404. Verify the AZURE_AI_PROJECT_ENDPOINT "
                "environment variable points to an existing project and that your account has access."
            ) from exc
        except HttpResponseError as exc:  # pragma: no cover - network failure guard
            if owns_credential:
                await credential.close()
            raise RuntimeError(
                f"Failed to ensure persistent Azure AI Foundry agent '{agent_name}': {exc}"
            ) from exc
        except Exception:
            if owns_credential:
                await credential.close()
            raise

        tools_list = list(tools) if tools else []
        if not tools_list:
            tools_argument: ToolLike | list[ToolLike] | None = None
        elif len(tools_list) == 1:
            tools_argument = cast(ToolLike, tools_list[0])
        else:
            tools_argument = cast(list[ToolLike], tools_list)

        agent_kwargs = dict(kwargs)
        run_kwargs: dict[str, Any] = {}
        if response_format is not None:
            run_kwargs["response_format"] = response_format

        tool_choice_default = agent_kwargs.pop("tool_choice", None)
        if tool_choice_default is None and tools_list:
            tool_choice_default = "auto"
        if tool_choice_default is not None:
            run_kwargs["tool_choice"] = tool_choice_default

        agent_kwargs.setdefault("model", llm_deployment_name)

        client = AzureAIAgentClient(
            project_endpoint=project_endpoint,
            model_deployment_name=llm_deployment_name,
            agent_id=persistent_agent_id,
            agent_name=agent_name,
            async_credential=credential,
        )

        llm_agent = ChatAgent(
            chat_client=client,
            name=agent_name,
            instructions=system_prompt,
            tools=tools_argument,
            **agent_kwargs,
        )

        self = cast("AFAFAgent", object.__new__(cls))
        self._credential = credential
        self._owns_credential = owns_credential
        self._client = client
        self.llm_agent = llm_agent
        self.agent_id = persistent_agent_id
        self._project_endpoint = project_endpoint
        self._model = llm_deployment_name
        self._system_prompt = system_prompt
        self._memory_max_tokens = int(memory_max_tokens)
        self._run_kwargs = run_kwargs
        self._history = []
        self._llm_api_key = llm_api_key

        self._encoding = None
        if _HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

        self._thread = self.llm_agent.get_new_thread()
        return self

    @staticmethod
    async def _find_agent_by_name(agents_client: Any, agent_name: str) -> Any | None:
        async for agent in agents_client.list_agents():
            if getattr(agent, "name", None) == agent_name:
                return agent
        return None

    async def aclose(self) -> None:
        await self._client.close()
        if self._owns_credential and hasattr(self._credential, "close"):
            await self._credential.close()  # type: ignore[func-returns-value]

    def reset_memory(self) -> None:
        self._thread = self.llm_agent.get_new_thread()
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
        total = 0
        for entry in self._history:
            total += self._estimate_tokens(entry.get("content", ""))
        return total

    def _prune_if_needed(self) -> None:
        if self._memory_max_tokens <= 0:
            return
        while self._current_token_count() > self._memory_max_tokens and len(self._history) > 1:
            removed = False
            for idx, msg in enumerate(self._history):
                if msg.get("role") != "system":
                    del self._history[idx]
                    removed = True
                    break
            if not removed:
                break

    def get_history(self, limit: int | None = None) -> List[dict[str, Any]]:
        if limit is not None and limit > 0:
            return self._history[-limit:]
        return list(self._history)

    def memory_stats(self) -> dict[str, Any]:
        total_tokens = self._current_token_count()
        messages = len(self.get_history())
        return {
            "messages": messages,
            "token_estimate": total_tokens,
            "token_limit": self._memory_max_tokens,
            "utilization_pct": round(100 * total_tokens / self._memory_max_tokens, 2)
            if self._memory_max_tokens
            else None,
        }

    async def __call__(self, user_prompt: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self._history.append({"role": "user", "content": user_prompt})

        merged_kwargs = {**self._run_kwargs, **kwargs}
        response = await self.llm_agent.run(
            messages=user_prompt,
            thread=self._thread,
            **merged_kwargs,
        )

        response_text = getattr(response, "text", None)
        if response_text is None and hasattr(response, "output"):
            response_text = getattr(getattr(response, "output"), "text", None)

        agent_response: dict[str, Any] = {
            "name": getattr(self.llm_agent, "name", "agent"),
            "content": response_text or "",
        }

        if agent_response["content"]:
            self._history.append({"role": "assistant", "content": agent_response["content"]})
            self._prune_if_needed()

        intermediate_messages: Dict[str, List[dict[str, Any]]] = {"function_calls": []}
        call_index: Dict[str, dict[str, Any]] = {}

        for message in getattr(response, "messages", []) or []:
            for content in message.contents:
                if isinstance(content, AFFunctionCallContent):
                    entry = {
                        "name": content.name,
                        "arguments": content.parse_arguments(),
                        "result": None,
                    }
                    if getattr(content, "call_id", None):
                        call_index[content.call_id] = entry
                    intermediate_messages["function_calls"].append(entry)
                elif isinstance(content, AFFunctionResultContent):
                    target = call_index.get(content.call_id)
                    if target is not None:
                        target["result"] = content.result
                    else:
                        intermediate_messages["function_calls"].append(
                            {"name": None, "arguments": None, "result": content.result}
                        )

        return agent_response, intermediate_messages