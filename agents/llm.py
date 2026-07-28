import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel
import asyncio

from langchain.tools import BaseTool
from typing import List, Optional

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
import httpx
from typing import Any, Sequence, Dict, Callable, Union

from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = float(os.getenv("RITS_REQUEST_TIMEOUT_SECONDS", 60.0))
MAX_RETRIES = int(os.getenv("RITS_MAX_RETRIES", 2))

timeout = httpx.Timeout(
    connect=10.0,
    read=REQUEST_TIMEOUT,
    write=30.0,
    pool=10.0,
)

class RITSChatModel(BaseChatModel):
    """LangChain-compatible chat model using httpx for internal RITS inference service."""

    # Mapping from endpoint name (short) to payload model name (full)
    MODEL_NAME_MAPPING: Dict[str, str] = {
        # Open Source Models
        "qwen3-5-397b-a17b-fp8": "Qwen/Qwen3.5-397B-A17B-FP8", 
        "mistral-large-3-675b-2512-fp4": "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
        "glm-5-1": "",
        "moonshotai-kimi-k2-5":"moonshotai/Kimi-K2.5",
        "gpt-oss-120b": "openai/gpt-oss-120b-a100",
        # smaller models
        "llama-3-3-70b-instruct": "meta-llama/llama-3-3-70b-instruct",
        "qwen2-5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    }

    model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    bound_tools: Optional[List[Dict[str, Any]]] = Field(default=None)

    @model_validator(mode="after")
    def _ping_endpoint(self) -> "RITSChatModel":
        url = f"{self.base_url}/{self.model_name}/ping"
        headers = {"RITS_API_KEY": self.api_key}
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"RITS ping failed for model '{self.model_name}' "
                f"(HTTP {e.response.status_code}). "
                f"Check that the model name is one of: "
                f"{list(self.MODEL_NAME_MAPPING.keys())}. "
                f"URL: {url}"
            )
            raise
        except httpx.TimeoutException:
            logger.error(
                f"RITS ping timed out for model '{self.model_name}' "
                f"at {url}. The model backend is likely not deployed "
                f"or is unreachable. Inference calls will also hang."
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"RITS ping failed for model '{self.model_name}': "
                f"could not connect to {url}. "
                f"Check that base_url is correct: {self.base_url}. "
                f"Error: {e}"
            )
            raise
        return self

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Convert LangChain messages to simple dicts
        msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                msgs.append({"role": "system", "content": m.content})

            elif isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})

            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    msg_dict["tool_calls"] = m.additional_kwargs.get("tool_calls")
                if "reasoning" in m.additional_kwargs:
                    msg_dict["reasoning"] = m.additional_kwargs["reasoning"]

                msgs.append(msg_dict)

            elif isinstance(m, ToolMessage):
                msgs.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content
                })
            
            else:
                # Fallback for unexpected types
                msgs.append({"role": "user", "content": m.content})

        # Use short name for endpoint URL
        url = f"{self.base_url}/{self.model_name}/v1/chat/completions"
        headers = {"RITS_API_KEY": self.api_key}

        # Use full name for payload if mapping exists, otherwise use model_name
        payload_model_name = self.MODEL_NAME_MAPPING.get(
            self.model_name,
            self.model_name
        )

        # Build request payload
        payload = {
            "model": payload_model_name,
            "messages": msgs,
            "temperature": self.temperature,
            **kwargs
        }

        # Include tools if bound
        if self.bound_tools:
            payload["tools"] = self.bound_tools

        # Add MAX_RETRIES and timeout handling
        # async with httpx.AsyncClient(timeout=timeout) as client:
        #     for attempt in range(MAX_RETRIES + 1):
        #         try:
        #             resp = await client.post(
        #                 url,
        #                 json=payload,
        #                 headers=headers,
        #             )
        #             resp.raise_for_status()
        #             break

        #         except httpx.ReadTimeout:
        #             if attempt == MAX_RETRIES:
        #                 raise
        #             await asyncio.sleep(2 ** attempt)

        #         except httpx.HTTPError:
        #             if attempt == MAX_RETRIES:
        #                 raise
        #             await asyncio.sleep(2 ** attempt)
        #         data = resp.json()

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=float(os.environ.get("RITS_REQUEST_TIMEOUT_SECONDS", "60"))
            )
            resp.raise_for_status()
            data = resp.json()

        # Handle OpenAI-like structure with optional tool calls
        msg_data = data["choices"][0]["message"]
        content = msg_data.get("content") or ""

        # Check for tool calls
        additional_kwargs = {}
        if "tool_calls" in msg_data:
            additional_kwargs["tool_calls"] = msg_data["tool_calls"]
        if "reasoning" in msg_data and msg_data["reasoning"]:
            additional_kwargs["reasoning"] = msg_data["reasoning"]

        message = AIMessage(
            content=content, additional_kwargs=additional_kwargs
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Synchronous wrapper for _agenerate."""
        try:
            asyncio.get_running_loop()
            # Event loop is running, we shouldn't be here
            raise RuntimeError(
                "Cannot call synchronous _generate from within async context. "
                "Use ainvoke() or agenerate() instead."
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self._agenerate(messages, stop, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "rits-openai-compat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs
    ) -> "RITSChatModel":
        """Bind tools to this chat model.

        Args:
            tools: List of tools to bind (MCPToolWrapper, dicts, etc.)
            **kwargs: Additional arguments to pass to model

        Returns:
            New instance of RITSChatModel with tools bound
        """
        from langchain_core.utils.function_calling import (
            convert_to_openai_tool
        )

        tool_defs = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "args"):
                # Build the tool definition manually 
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.args,
                            "required": list(tool.args.keys())
                        }
                    }
                }
                tool_defs.append(tool_def)
            else:
                try:
                    # Try LangChain's standard conversion
                    tool_defs.append(convert_to_openai_tool(tool))
                except Exception as e:
                    # Fallback: if tool has schema/dict representation
                    if isinstance(tool, dict):
                        tool_defs.append(tool)
                    else:
                        raise ValueError(
                            f"Unable to convert tool {tool} to OpenAI format: {e}"
                        )

        # Create new instance with tools bound
        return self.model_copy(
            update={"bound_tools": tool_defs, **kwargs}
        )


def create_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0,
    **kwargs,
) -> BaseChatModel:
    """Create LLM instance based on provider.

    Args:
        provider: One of "rits", "ollama", "anthropic", "openai", "litellm", "watsonx"
        model: Model name (optional, uses defaults if not provided)
        api_key: API key (optional, falls back to environment variables)
        temperature: Sampling temperature (default: 0)
        **kwargs: Provider-specific arguments:
            - ollama: ollama_base_url (str)
            - litellm: api_base (str)
            - watsonx: project_id (str), space_id (str)

    Returns:
        LangChain BaseChatModel instance
    """
    if provider == "watsonx":
        kwargs.setdefault("project_id", os.environ.get("WATSONX_PROJECT_ID"))
        kwargs.setdefault("space_id", os.environ.get("WATSONX_SPACE_ID"))
        if not api_key:
            api_key = os.environ.get("WATSONX_APIKEY")
    elif provider == "litellm":
        resolved_base_url = os.environ.get("LITELLM_BASE_URL")
        if resolved_base_url:
            kwargs.setdefault("api_base", resolved_base_url)
        if not api_key:
            api_key = os.environ.get("LITELLM_API_KEY")
    if provider == "rits":
        rits_api_key = api_key or os.environ.get("RITS_API_KEY")
        if not rits_api_key:
            raise ValueError(
                "You need to set the env var RITS_API_KEY to use a "
                "model from RITS."
            )

        base_url = (
            "https://inference-3scale-apicast-production.apps.rits."
            "fmaas.res.ibm.com"
        )
        model_name = model or "llama-3-3-70b-instruct"

        return RITSChatModel(
            model_name=model_name,
            base_url=base_url,
            api_key=rits_api_key,
            temperature=temperature,
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama support. "
                "Install with: pip install langchain-ollama"
            )

        model_name = model or "llama3.1:8b"
        base_url = (
            kwargs.get("ollama_base_url")
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

        print(f"Connecting to Ollama at {base_url}")
        print(
            f"Make sure Ollama is running and the model "
            f"'{model_name}' is pulled."
        )
        print("  To start Ollama: ollama serve")
        print(f"  To pull model: ollama pull {model_name}\n")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=65536,
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic support. "
                "Install with: pip install langchain-anthropic"
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "You need to set the env var ANTHROPIC_API_KEY to use "
                "an Anthropic model."
            )

        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            temperature=temperature,
            api_key=resolved_key,
        )

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI support. "
                "Install with: pip install langchain-openai"
            )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "You need to set the env var OPENAI_API_KEY to use "
                "an OpenAI model."
            )

        return ChatOpenAI(
            model=model or "gpt-4.1",
            temperature=temperature,
            api_key=resolved_key,
        )

    elif provider == "litellm":
        try:
            from langchain_litellm import ChatLiteLLM
            # Uncomment to debug litellm connection
            # import litellm
            # litellm._turn_on_debug()
        except ImportError:
            raise ImportError(
                "langchain-litellm is required for LiteLLM support. "
                "Install with: pip install langchain-litellm"
            )

        # This is the cheapest model in our allowed set: 
        # ['aws/claude-opus-4-5', 'claude-opus-4-5-20251101', 
        # 'claude-sonnet-4-5-20250929', 'aws/claude-sonnet-4-5', 
        # 'Azure/gpt-5.1-2025-11-13', 'GCP/gemini-2.5-flash', 
        # 'GCP/gemini-2.5-flash-lite', 'GCP/gemini-2.0-flash', 
        # 'gcp/gemini-3-flash-preview']
        model_name = model or "GCP/gemini-2.0-flash"
        params: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            params["api_key"] = api_key
        if "api_base" in kwargs:
            params["api_base"] = kwargs["api_base"]
        # This parameter is critical. Without it, the client attempts
        # to infer the provider name from the base of the model name, 
        # and there is no way to satisfy the check for the model being in 
        # the allow-list [e.g. GCP/gemini-2.0-flash] and the check for an
        # existing provider (GCP isn't an existing provider)
        params["custom_llm_provider"] = "openai"

        return ChatLiteLLM(**params)

    elif provider == "watsonx":
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError:
            raise ImportError(
                "langchain-ibm is required for watsonx support. "
                "Install with: pip install langchain-ibm"
            )

        resolved_key = api_key or os.environ.get("WATSONX_APIKEY")
        if not resolved_key:
            raise ValueError(
                "You need to set the env var WATSONX_APIKEY to use "
                "a watsonx.ai model."
            )

        project_id = kwargs.get("project_id") or os.environ.get("WATSONX_PROJECT_ID")
        space_id = kwargs.get("space_id") or os.environ.get("WATSONX_SPACE_ID")

        if not project_id and not space_id:
            raise ValueError(
                "Either project_id or space_id is required for watsonx.ai. "
                "Set WATSONX_PROJECT_ID or WATSONX_SPACE_ID environment variable."
            )

        url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        params: Dict[str, Any] = {
            "model_id": model or "openai/gpt-oss-120b",
            "url": url,
            "apikey": resolved_key,
            "params": {
                "temperature": temperature,
                "max_new_tokens": 4096,
            },
        }

        if project_id:
            params["project_id"] = project_id
        elif space_id:
            params["space_id"] = space_id

        return ChatWatsonx(**params)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Must be one of: 'rits', 'ollama', 'anthropic', 'openai', "
            "'litellm', 'watsonx'"
        )
