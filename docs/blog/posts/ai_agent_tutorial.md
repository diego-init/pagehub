---
title: "AI Agents"
date: 2025-02-13
authors:
  - diego-init
categories:
  - agent
tags:
  - AI
  - agents
  - LLM
  - automation
comments: true
---

# Getting Started with AI Agents



## What’s an AI Agent, Anyway?

An **AI Agent** is just a smart system that uses AI to interact with its surroundings and get stuff done. It thinks, plans, and takes action (sometimes using extra tools) to complete tasks.

AI Agents can do anything we set them up for using **Tools** to carry out **Actions**.

### The Two Main Parts of an AI Agent

1. **The Brain (AI Model)** - This is where all the decision-making happens. The AI figures out what to do next. Examples include Large Language Models (LLMs) like GPT-4.

2. **The Body (Tools & Capabilities)** - This is what the agent actually *does*. Its abilities depend on the tools it has access to.

## Why Use LLMs?

LLMs (Large Language Models) are the go-to choice for AI Agents because they’re great at understanding and generating text. Popular ones include GPT-4, Llama, and Gemini.

There are two ways you can use an LLM:

- **Run Locally** (if your computer is powerful enough).
- **Use a Cloud/API** (e.g., via Hugging Face’s API).

## System Messages: Setting the Rules

System messages (or prompts) tell the AI how it should behave. They act as guiding instructions.

```python
system_message = {
    "role": "system",
    "content": "You are a helpful customer service agent. Always be polite and clear."
}
```

These messages also define what tools the AI can use and how it should format its responses.

## Conversations: How AI Talks to Users

A conversation is just back-and-forth messages between a user and the AI. Chat templates help keep things organized and make sure the AI remembers what’s going on.

Example:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "Sure! What’s your order number?"},
    {"role": "user", "content": "ORDER-123"},
]
```

## Chat Templates: Keeping AI Conversations Structured

Chat templates make sure LLMs correctly process messages. There are two main types of AI models:

- **Base Models**: Trained on raw text to predict the next word.
- **Instruct Models**: Fine-tuned to follow instructions and have conversations.

We use **ChatML**, a structured format for messages. The transformers library takes care of this automatically:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## What are Tools?

Tools let AI Agents do more than just text generation. A Tool is basically a function the AI can call to get things done.

| Tool             | What It Does                                        |
| ---------------- | --------------------------------------------------- |
| Web Search       | Fetches up-to-date info from the internet.          |
| Image Generation | Creates images from text.                           |
| Retrieval        | Pulls in data from other sources.                   |
| API Interface    | Connects with external APIs like GitHub or Spotify. |

### Why Do AI Agents Need Tools?

LLMs have a limited knowledge base (they only know what they were trained on). Tools help by allowing:

- **Real-time data fetching** (e.g., checking the weather).
- **Specialized tasks** (e.g., doing math, calling APIs).

## Building a Simple Tool: A Calculator

Let’s create a basic calculator tool that multiplies two numbers:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

This tool includes:

- A clear name (`calculator`).
- A description (via the docstring).
- Input and output types.

To define it as a tool, we describe it like this:

```python
Tool Name: calculator, Description: Multiplies two numbers., Arguments: a: int, b: int, Outputs: int
```

### Automating Tool Descriptions

Instead of writing descriptions manually, we can use Python introspection to extract details automatically. The `Tool` class helps manage this info.

```python
class Tool:
    def __init__(self, name: str, description: str, func: callable, arguments: list, outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        args_str = ", ".join([f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments])
        return f"Tool Name: {self.name}, Description: {self.description}, Arguments: {args_str}, Outputs: {self.outputs}"

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
```

Now, we can create a tool instance:

```python
calculator_tool = Tool(
    "calculator", "Multiplies two numbers.", calculator, [("a", "int"), ("b", "int")], "int"
)
```

### Using a Decorator to Define Tools

A decorator makes tool creation easier:

```python
import inspect

def tool(func):
    signature = inspect.signature(func)
    arguments = [(param.name, param.annotation.__name__) for param in signature.parameters.values()]
    return_annotation = signature.return_annotation.__name__ if signature.return_annotation else "No return annotation"
    return Tool(func.__name__, func.__doc__ or "No description provided.", func, arguments, return_annotation)
```

Now, we can define tools like this:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

This makes it easy for AI Agents to recognize and use tools based on text input.

## Recap

- **AI Agents** use AI models to interact and make decisions.
- **LLMs** handle language understanding and text generation.
- **System Messages** define the agent’s behavior.
- **Tools** extend an AI’s capabilities beyond text generation.
- **Chat Templates** format conversations properly.
- **Tools** help AI Agents fetch real-time data and execute tasks.

By combining all these pieces, you can build smart AI Agents that think, act, and assist like pros!
