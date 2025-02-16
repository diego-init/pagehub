---
title: "Getting Started with AI Agents"
date: 2025-02-13
authors:
  - diego-init
categories:
  - AI Agents
tags:
  - AI
  - agents
  - LLM
  - automation
comments: true
---

# Getting Started with AI Agents - Part I

It's hard to have a conversation about AI these days without bringing up tools like ChatGPT, DeepSeek, and the like, right? These tools are becoming key players in how we analyze and interact with public transport. Are you ready for this shift?

In this post, let's chat a bit about AI Agents and how we can tailor LLMs to answer the questions that matter most to us as Transit Data Scientists.

Before diving into more complex applications, let’s take a moment to introduce the topic. This way, we can get comfortable with the terms and concepts we'll be working with!

This tutorial is a summmary of [HuggingFace](https://huggingface.co/learn/agents-course/unit0/introduction) Course on Agents - Unit 1.

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
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: callable, 
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool, 
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```

Now, we can create a tool instance:

```python
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)
```

### Using a Decorator to Define Tools

A decorator makes tool creation easier:

```python
def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)
    
    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__ 
            if hasattr(param.annotation, '__name__') 
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))
    
    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__ 
            if hasattr(return_annotation, '__name__') 
            else str(return_annotation)
        )
    
    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."
    
    # The function name becomes the Tool name
    name = func.__name__
    
    # Return a new Tool instance
    return Tool(
        name=name, 
        description=description, 
        func=func, 
        arguments=arguments, 
        outputs=outputs
    )
```

Now, we can define tools like this:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
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

In the next tutorial we will discuss the AI ​​Agents workflow.
