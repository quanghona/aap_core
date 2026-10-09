# aap_core

Agent design pattern core framework for building, orchestrating, and training AI agents.

![PyPI - License](https://img.shields.io/pypi/l/aap_core)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aap_core)
![PyPI - Version](https://img.shields.io/pypi/v/aap_core?label=%20)
![Python Version](https://img.shields.io/pypi/pyversions/aap_core)

## What is this?

`aap_core` is the foundational framework within the **AI Agent Pattern** project. It provides a unified abstraction layer for building AI agents, orchestrating multi-agent workflows, and training policies via reinforcement learning — all built on Pydantic and compatible with the [A2A protocol](https://github.com/google/a2a-sdk).

The framework separates orchestration logic from LLM manipulation, delegating model interaction to integration packages (`aap_langchain`, `aap_llamaindex`, `aap_dspy`, `aap_transformers`). This lets you swap backends without changing agent architecture.

## Quick Start

```bash
pip install aap_core
# or
uv add aap_core
```

```python
from a2a.types import AgentCard
from aap_core.agent import BaseAgent
from aap_core.types import AgentMessage

# Define a minimal agent
class MyAgent(BaseAgent):
    card: AgentCard = AgentCard(
        name="my-agent",
        description="A simple agent",
        skills=[],
    )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Your agent logic here
        message.responses = [("my-agent", "Hello!")]
        message.execution_result = "success"
        return message
```

## Features

- **Agent Abstraction**: Clean `BaseAgent` interface with A2A `AgentCard` integration for self-describing agent metadata
- **Orchestration Patterns**: Built-in agents for reflection, looping, voting, coordination, and debate patterns
- **Chain System**: Generic multi-turn causal chains with history management, tool processing, and token tracking
- **Prompt Augmentation**: Data and structure-based prompt enhancement with configurable looping strategies
- **Retrieval System**: Pluggable retrievers with post-processing (e.g., rerankers, summarizers)
- **RL Policy Training**: Full RL loop with replay buffers, exploration modules, and trainers (PPO, DPO, GRPO, Reinforce++), powered by Gymnasium
- **Thought Chains**: Chain-of-Thought, Tree-of-Thought, Graph-of-Thought, and Self-Consistency patterns
- **Guardrails**: Extensible guardrail system for input/output filtering
- **Async Support**: All agent execution methods have async counterparts (`aexecute`)
- **Token Usage Tracking**: Automatic token accounting across multi-agent chains

## Installation

### Core

```bash
pip install aap_core
```

### Integration Packages

```bash
pip install aap_langchain   # LangChain integration
pip install aap_llamaindex  # LlamaIndex integration
pip install aap_dspy        # DSPy integration
pip install aap_transformers  # Hugging Face Transformers integration
```

## Usage

### Building an Agent

```python
from aap_core.agent import BaseAgent
from aap_core.types import AgentMessage
from a2a.types import AgentCard

class TaskAgent(BaseAgent):
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        # Process the message...
        message.responses = [(self.card.name, "Task completed")]
        message.execution_result = "success"
        self.state = "idle"
        return message
```

### Using Orchestration Patterns

```python
from aap_core.orchestration import ReflectionAgent, LoopAgent

# Self-reflection: execute then reflect on the result
reflector = ReflectionAgent(
    chain_task=my_chain,
    chain_reflection=reflection_chain,
)
result = reflector.execute(message)

# Loop until a condition is met
stopper = lambda msg: "done" in msg.query.lower()
loop = LoopAgent(agent=my_agent, stop_condition=stopper)
result = loop.execute(message)
```

### Prompt Augmentation

```python
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Augment prompts with external context data
augmenter = SimplePromptAugmenter(
    format="Context:\n{context}\n\nQuery: {query}",
    loop=3,  # Apply 3 times
)
augmented = augmenter(message)
```

### RL-Based Prompt Optimization

```python
from aap_core.policy_trainer import PPOTrainer, SimpleReplayBuffer
from aap_core.prompt_augmenter import PromptOptimizationEnv

# Create the environment with multiple augmenters
env = PromptOptimizationEnv(augmenters=[augmenter_a, augmenter_b])

# Create a replay buffer and trainer
buffer = SimpleReplayBuffer(capacity=10000)
trainer = PPOTrainer(policy=my_policy, buffer=buffer)
trainer.train(env, num_steps=10000)
```

## Architecture

`aap_core` is organized into the following modules:

| Module | Description |
|--------|-------------|
| `agent` | Base agent interface with A2A `AgentCard` support and state management |
| `orchestration` | Multi-agent patterns: reflection, looping, voting, coordination, debate |
| `chain` | Generic multi-turn causal chains with history, tool processing, and token tracking |
| `policy` | RL policy models (GPT-2 based) with RMSNorm, RoPE, and GQA |
| `policy_trainer` | Replay buffers, exploration modules, and trainers (PPO, DPO, GRPO, Reinforce++) |
| `prompt_augmenter` | Data and structure-based prompt enhancement with Gymnasium environments |
| `retriever` | Pluggable retrieval system with post-processing support |
| `thought` | Thought chain patterns: CoT, ToT, GoT, Self-Consistency |
| `guardrail` | Input/output filtering system |
| `types` | Core type definitions: `AgentMessage`, `TokenUsage`, `BaseLLMChain`, `BaseChain` |
| `utils` | Utility functions (e.g., thinking tag removal) |

## Documentation

- Full project documentation: [aap_core docs](https://github.com/quanghona/aap_core)
- Example notebooks: [example](https://github.com/quanghona/aap_example.git)
- A2A Protocol: [google/a2a-sdk](https://github.com/google/a2a-sdk)

## Contributing

See the [main project README](https://github.com/quanghona/aap_core) for contribution guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/quanghona/aap_core/blob/master/LICENSE) file for details.
