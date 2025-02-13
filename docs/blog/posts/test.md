---
title: "teste"
date: 2025-02-03
authors:
  - diego-init
categories:
  - welcome
tags:
  - Foo
  - Bar
draft: true

comments: true
---

# What is an Agent?

<blockquote>
An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.
</blockquote>

An Agent can perform any task we implement via **Tools** to complete **Actions**.

Think of the Agent as having two main parts:

* **The Brain (AI Model)** - This is where all the thinking happens. The AI model handles reasoning and planning. It decides which Actions to take based on the situation. For example, LLMs.

* **The Body (Capabilities and Tools)** - This part represents everything the Agent is equipped to do. The scope of possible actions depends on what the agent has been equipped with.

# Why LLM?

The most common AI model found in Agents is an LLM (Large Language Model), which takes Text as an input and outputs Text as well. An LLM is a type of AI model that excels at understanding and generating human language.
The underlying principle of an LLM is simple yet highly effective: its objective is to predict the next token, given a sequence of previous tokens. A “token” is the unit of information an LLM works with. You can think of a “token” as if it was a “word”, but for efficiency reasons LLMs don’t use whole words.
Well known examples are GPT4 from OpenAI, LLama from Meta, Gemini from Google, etc. These models have been trained on a vast amount of text and are able to generalize well. It's also possible to use models that accept other inputs as the Agent's core model. For example, a Vision Language Model (VLM), which is like an LLM but also understands images as input. 

You have two main options:

* Run Locally (if you have sufficient hardware).

* Use a Cloud/API (e.g., via the Hugging Face Serverless Inference API).

LLMs are a key component of AI Agents, providing the foundation for understanding and generating human language.

##  System Messages: The Underlying System of LLMs

System messages (also called System Prompts) define how the model should behave. They serve as persistent instructions, guiding every subsequent interaction.

For example:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```
When using Agents, the System Message also gives information about the available tools, provides instructions to the model on how to format the actions to take, and includes guidelines on how the thought process should be segmented.

## Conversations: User and Assistant Messages

A conversation consists of alternating messages between a Human (user) and an LLM (assistant).

Chat templates help maintain context by preserving conversation history, storing previous exchanges between the user and the assistant. This leads to more coherent multi-turn conversations.

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

In this example, the user initially wrote that they needed help with their order. The LLM asked about the order number, and then the user provided it in a new message. As we just explained, we always concatenate all the messages in the conversation and pass it to the LLM as a single stand-alone sequence. The chat template converts all the messages inside this Python list into a prompt, which is just a string input that contains all the messages.

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

## Chat-Templates

Chat templates are essential for structuring conversations between language models and users. They guide how message exchanges are formatted into a single prompt.

Another point we need to understand is the difference between a Base Model vs. an Instruct Model:

* A **Base Model** is trained on raw text data to predict the next token.

* An **Instruct Model** is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant). If you have interacted with some AI API lately, you know that’s the standard practice.

It’s important to note that a base model could be fine-tuned on different chat templates, so when we’re using an instruct model we need to make sure we’re using the correct chat template.

In transformers, chat templates include Jinja2 code that describes how to transform the ChatML list of JSON messages, as presented in the above examples, into a textual representation of the system-level instructions, user messages and assistant responses that the model can understand.

The transformers library will take care of chat templates for you as part of the tokenization process. For more info,click here <https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates>

The easiest way to ensure your LLM receives a conversation correctly formatted is to use the chat_template from the model’s tokenizer.

To convert the previous conversation into a prompt, we load the tokenizer and call apply_chat_template:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This <code>apply_chat_template()</code> function will be used in the backend of your API, when you interact with messages in the ChatML format.

#  What are Tools?

One crucial aspect of AI Agents is their ability to take actions. As we saw, this happens through the use of Tools.

A Tool is a function given to the LLM. This function should fulfill a clear objective.

|Tool 	|Description|
| -------- | ------- |
|Web Search 	|Allows the agent to fetch up-to-date information from the internet.|
|Image Generation 	|Creates images based on text descriptions.|
|Retrieval 	|Retrieves information from an external source.|
|API Interface 	|Interacts with an external API (GitHub, YouTube, Spotify, etc.).|

A good tool should be something that complements the power of an LLM.

For instance, if you need to perform arithmetic, giving a calculator tool to your LLM will provide better results than relying on the native capabilities of the model.

Furthermore, LLMs predict the completion of a prompt based on their training data, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.

A Tool should contain:

* A textual description of what the function does.
* A Callable (something to perform an action).
* Arguments with typings.
* (Optional) Outputs with typings.

LLMs, as we saw, can only receive text inputs and generate text outputs. They have no way to call tools on their own. What we mean when we talk about providing tools to an Agent, is that we teach the LLM about the existence of tools, and ask the model to generate text that will invoke tools when it needs to. For example, if we provide a tool to check the weather at a location from the Internet, and then ask the LLM about the weather in Paris, the LLM will recognize that question as a relevant opportunity to use the “weather” tool we taught it about. The LLM will generate text, in the form of code, to invoke that tool. It is the responsibility of the Agent to parse the LLM’s output, recognize that a tool call is required, and invoke the tool on the LLM’s behalf. The output from the tool will then be sent back to the LLM, which will compose its final response for the user.

## How do we give tools to an LLM?

We will implement a simplified calculator tool that will just multiply two integers. This could be our Python implementation:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

So our tool is called calculator, it multiplies two integers, and it requires the following inputs:

    a (int): An integer.
    b (int): An integer.

The output of the tool is another integer number that we can describe like this:

* (int): The product of a and b.

All of these details are important. Let’s put them together in a text string that describes our tool for the LLM to understand.

```python
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

Our implementation already provides everything we need:

* A descriptive name of what it does: calculator
* A longer description, provided by the function’s docstring comment: Multiply two integers.
* The inputs and their type: the function clearly expects two ints.
* The type of the output.

We could provide the Python source code as the specification of the tool for the LLM, but the way the tool is implemented does not matter. All that matters is its name, what it does, the inputs it expects and the output it provides.

We will leverage Python’s introspection features to leverage the source code and build a tool description automatically for us. All we need is that the tool implementation uses type hints, docstrings, and sensible function names. We will write some code to extract the relevant portions from the source code.

After we are done, we’ll only need to use a Python decorator to indicate that the calculator function is a tool:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

```python
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

As you can see, it’s the same thing we wrote manually before!

##  Generic Tool implementation

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

It may seem complicated, but if we go slowly through it we can see what it does. We define a Tool class that includes:

<code>name (str)</code>: The name of the tool.
<code>description (str)</code>: A brief description of what the tool does.
<code>function (callable)</code>: The function the tool executes.
<code>input_arguments (list)</code>: The expected input parameters.
<code>outputs (str or list)</code>: The expected outputs of the tool.
<code>__call__()</code>: Calls the function when the tool instance is invoked.
<code>to_string()</code>: Converts the tool’s attributes into a textual representation.

We could create a Tool with this class using code like the following:

```python
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)
```

The decorator implementation:

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

With this decorator in place we can implement our tool like this:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

And we can use the Tool’s to_string method to automatically retrieve a text suitable to be used as a tool description for an LLM:

```python
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

To summarize, we learned:

* What Tools Are: Functions that give LLMs extra capabilities, such as performing calculations or accessing external data.

* How to Define a Tool: By providing a clear textual description, inputs, outputs, and a callable function.

* Why Tools Are Essential: They enable Agents to overcome the limitations of static model training, handle real-time tasks, and perform specialized actions.