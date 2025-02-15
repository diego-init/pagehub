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

# Getting Started with AI Agents - Part II

In previous sections, we learned how tools are provided to agents in the system prompt and how AI agents can reason, plan, and interact with their environment.

Now, we'll examine the AI Agent Workflow, known as Thought-Action-Observation.

This tutorial is inspired by [HuggingFace](https://huggingface.co/learn/agents-course/unit0/introduction).

##  The Thought-Action-Observation Cycle

```mermaid
st=>start: Start
op=>operation: Your Operation
cond=>condition: Yes or No?
e=>end

st->op->cond
cond(yes)->e
cond(no)->op
```
