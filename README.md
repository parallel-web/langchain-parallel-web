# LangChain Parallel Web Integration

This package provides LangChain integrations for [Parallel AI](https://docs.parallel.ai/), enabling real-time web research and AI capabilities through an OpenAI-compatible interface.

## Features

- **Chat Models**: `ChatParallelWeb` - Real-time web research chat completions
- **Search Tools**: `ParallelWebSearchTool` - Direct access to Parallel AI's Search API
- **Extract Tools**: `ParallelExtractTool` - Clean content extraction from web pages
- **Streaming Support**: Real-time response streaming
- **Async/Await**: Full asynchronous operation support
- **OpenAI Compatible**: Uses familiar OpenAI SDK patterns
- **LangChain Integration**: Seamless integration with LangChain ecosystem

## Installation

```bash
pip install langchain-parallel-web
```

## Setup

1. Get your API key from [Parallel AI](https://parallel.ai/)
2. Set your API key as an environment variable:

```bash
export PARALLEL_AI_API_KEY="your-api-key-here"
```

## Chat Models

### ChatParallelWeb

The `ChatParallelWeb` class provides access to Parallel AI's Chat API, which combines language models with real-time web research capabilities.

#### Basic Usage

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_parallel_web.chat_models import ChatParallelWeb

# Initialize the chat model
chat = ChatParallelWeb(
    model="speed",  # Parallel AI's chat model
    temperature=0.7,  # Optional: ignored by Parallel AI
    max_tokens=None,  # Optional: ignored by Parallel AI
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant with access to real-time web information."),
    HumanMessage(content="What are the latest developments in artificial intelligence?")
]

# Get response
response = chat.invoke(messages)
print(response.content)
```

#### Streaming Responses

```python
# Stream responses for real-time output
for chunk in chat.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

#### Async Operations

```python
import asyncio

async def main():
    # Async invoke
    response = await chat.ainvoke(messages)
    print(response.content)
    
    # Async streaming
    async for chunk in chat.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

#### Conversation Context

```python
# Maintain conversation history
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

# First turn
messages.append(HumanMessage(content="What is machine learning?"))
response = chat.invoke(messages)
messages.append(response)  # Add assistant response

# Second turn with context
messages.append(HumanMessage(content="How does it work?"))
response = chat.invoke(messages)
print(response.content)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"speed"` | Parallel AI model name |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses `PARALLEL_AI_API_KEY` env var if not provided) |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL |
| `temperature` | `Optional[float]` | `None` | Sampling temperature (ignored by Parallel AI) |
| `max_tokens` | `Optional[int]` | `None` | Max tokens (ignored by Parallel AI) |
| `timeout` | `Optional[float]` | `None` | Request timeout |
| `max_retries` | `int` | `2` | Max retry attempts |


## Real-Time Web Research

Parallel AI's Chat API provides real-time access to web information, making it perfect for:

- **Current Events**: Get up-to-date information about recent events
- **Market Data**: Access current stock prices, market trends
- **Research**: Find the latest research papers, developments
- **Weather**: Get current weather conditions
- **News**: Access breaking news and recent articles

```python
# Example: Current events
messages = [
    SystemMessage(content="You are a research assistant with access to real-time web data."),
    HumanMessage(content="What happened in the stock market today?")
]

response = chat.invoke(messages)
print(response.content)  # Gets real-time market information
```

## Integration with LangChain

### Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant with access to real-time web information."),
    ("human", "{question}")
])

chain = prompt | chat | StrOutputParser()

# Use the chain
result = chain.invoke({"question": "What are the latest AI breakthroughs?"})
print(result)
```

### Agents

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Create an agent with web research capabilities
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to real-time web information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Use with tools for additional capabilities
# agent = create_openai_functions_agent(chat, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
```

## Search API

The Search API provides direct access to Parallel AI's web search capabilities, returning structured, compressed excerpts optimized for LLM consumption.

### ParallelWebSearchTool

The search tool provides direct access to Parallel AI's Search API:

```python
from langchain_parallel_web import ParallelWebSearchTool

# Initialize the search tool
search_tool = ParallelWebSearchTool()

# Search with an objective
result = search_tool.invoke({
    "objective": "What are the latest developments in renewable energy?",
    "processor": "base",  # "base" for speed, "pro" for quality
    "max_results": 5
})

print(result)
# {
#     "search_id": "search_123...",
#     "results": [
#         {
#             "url": "https://example.com/renewable-energy",
#             "title": "Latest Renewable Energy Developments",
#             "excerpts": [
#                 "Solar energy has seen remarkable growth...",
#                 "Wind power capacity increased by 15%..."
#             ]
#         }
#     ]
# }
```





### Search API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | `Optional[str]` | `None` | Natural-language description of research goal |
| `search_queries` | `Optional[List[str]]` | `None` | Specific search queries (max 5, 200 chars each) |
| `processor` | `Literal["base", "pro"]` | `"base"` | Processing tier: "base" (fast) or "pro" (quality) |
| `max_results` | `int` | `10` | Maximum results to return (1-40) |
| `max_chars_per_result` | `int` | `1500` | Maximum characters per result (min 100) |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses env var if not provided) |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL |

### Processor Comparison

| Processor | Speed | Cost | Quality | Use Case |
|-----------|-------|------|---------|----------|
| **base** | 4-5s | Lower | Good | Real-time applications, quick searches |
| **pro** | 45-70s | Higher | Excellent | Research, analysis |

### Search with Specific Queries

You can provide specific search queries instead of an objective:

```python
# Search with specific queries
result = search_tool.invoke({
    "search_queries": [
        "renewable energy 2024",
        "solar power developments",
        "wind energy statistics"
    ],
    "processor": "pro",
    "max_results": 8
})
```

### Tool Usage in Agents

The search tool works seamlessly with LangChain agents:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Create agent with search capabilities
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use the search tool to find current information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
result = agent_executor.invoke({
    "input": "What are the latest developments in artificial intelligence?"
})
print(result["output"])
```

## Extract API

The Extract API provides clean content extraction from web pages, returning structured markdown-formatted content optimized for LLM consumption.

### ParallelExtractTool

The extract tool extracts clean, structured content from web pages:

```python
from langchain_parallel_web import ParallelExtractTool

# Initialize the extract tool
extract_tool = ParallelExtractTool()

# Extract from a single URL
result = extract_tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
})

print(result)
# [
#     {
#         "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
#         "title": "Artificial intelligence - Wikipedia",
#         "content": "# Artificial intelligence\n\nMain content in markdown...",
#         "publish_date": "2024-01-15"  # Optional
#     }
# ]
```

### Extract from Multiple URLs

```python
# Extract from multiple URLs in batch
result = extract_tool.invoke({
    "urls": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]
})

for item in result:
    print(f"Title: {item['title']}")
    print(f"URL: {item['url']}")
    print(f"Content length: {len(item['content'])} characters")
    print()
```

### Content Length Control

```python
# Control content length per extraction
extract_tool = ParallelExtractTool(max_chars_per_extract=2000)

result = extract_tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Quantum_computing"]
})

print(f"Content length: {len(result[0]['content'])} characters")
```

### Extract API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `List[str]` | Required | List of URLs to extract content from |
| `max_chars_per_extract` | `Optional[int]` | `None` | Maximum characters per extraction |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses env var if not provided) |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL |

### Error Handling

The extract tool gracefully handles failed extractions:

```python
# Mix of valid and invalid URLs
result = extract_tool.invoke({
    "urls": [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://this-domain-does-not-exist-12345.com/"
    ]
})

for item in result:
    if "error_type" in item:
        print(f"Failed: {item['url']} - {item['content']}")
    else:
        print(f"Success: {item['url']} - {len(item['content'])} chars")
```

### Async Extract

```python
import asyncio

async def extract_async():
    result = await extract_tool.ainvoke({
        "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
    })
    return result

# Run async extraction
result = asyncio.run(extract_async())
```

## Error Handling

```python
try:
    response = chat.invoke(messages)
    print(response.content)
except ValueError as e:
    if "API key not found" in str(e):
        print("Please set your PARALLEL_AI_API_KEY environment variable")
    else:
        print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

See the `examples/` and `docs/` directories for complete working examples:

- `examples/chat_example.py` - Chat model usage examples
- `docs/search_tool.ipynb` - Search tool examples and tutorials
- `docs/extract_tool.ipynb` - Extract tool examples and tutorials
- Basic synchronous usage
- Streaming responses
- Async operations
- Conversation management
- Tool usage in agents

## API Compatibility

This integration provides access to two Parallel AI APIs:

### Chat API Compatibility
The Chat API uses Parallel AI's OpenAI-compatible interface:

- **Supported**: Messages, streaming, response_format (JSON schema)
- **Ignored**: temperature, max_tokens, top_p, stop, most OpenAI-specific parameters
- **Not Supported**: Function calling, multimodal inputs (images/audio), tool usage

### Search API Features
The Search API provides direct web search capabilities:

- **Supported**: Objective-based search, query-based search, two processor tiers
- **Output**: Structured results with URLs, titles, and relevant excerpts
- **Integration**: Works with LangChain tools, retrievers, and agents

### Extract API Features
The Extract API provides clean content extraction from web pages:

- **Supported**: Batch URL extraction, content length control, markdown formatting
- **Output**: Clean, structured content with metadata (title, publish date, etc.)
- **Integration**: Works with LangChain tools and agents
- **Error Handling**: Gracefully handles failed extractions with detailed error info

## Performance & Rate Limits

### Chat API
- **Default Rate Limit**: 300 requests per minute
- **Performance**: 3 second p50 TTFT (time to first token) with streaming
- **Use Cases**: Interactive chat, real-time responses

### Search API
The Search API offers two processor tiers with different performance characteristics:

| Processor | p90 Latency | Cost ($/1000) | Max Results | Best For |
|-----------|-------------|---------------|-------------|----------|
| **base** | 4-5s | $4 | 40 | Fast searches, real-time applications |
| **pro** | 45-70s | $9 | 40 | Research, comprehensive analysis |

### Production Usage
Contact [Parallel AI](https://parallel.ai/) for:
- Higher rate limits
- Enterprise features
- Custom processor configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

### Documentation
- [Parallel AI Documentation](https://docs.parallel.ai/)
- [Chat API Reference](https://docs.parallel.ai/chat-api)
- [Search API Reference](https://docs.parallel.ai/search-api)
- [LangChain Documentation](https://python.langchain.com/)

### Getting Help
- [GitHub Issues](https://github.com/parallel-web/langchain-parallel-web/issues)
- [Parallel AI Support](mailto:support@parallel.ai)

## Changelog

### v0.1.0
- Initial release
- **Chat Models**: ChatParallelWeb with real-time web research
- **Search Tools**: ParallelWebSearchTool for direct API access
- **Extract Tools**: ParallelExtractTool for clean content extraction
- Streaming and async/await support
- Two processor tiers (base/pro) for search
- Batch URL extraction with error handling
- Full LangChain ecosystem compatibility
