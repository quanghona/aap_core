import concurrent.futures
import random
import re
from collections.abc import Generator, Iterator, Sequence
from typing import Callable, List, Literal, Tuple

from pydantic import Field, field_validator
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu

from aap_core.types import AgentResponse

from .types import AgentMessage, BaseLLMChain
from .agent import BaseAgent


class ReflectionAgent(BaseAgent):
    """
    An agent that is capable of self-reflection.

    This agent will first execute the given message and then reflect on the
    result. If the reflection is successful, the agent will return a new
    message with the reflection result. If the reflection is not successful,
    the agent will return the original message.

    Attributes:
        chain: The BaseLLMChain to use for executing the message.
        chain_reflection: The prompt to use for self-reflection.
    """

    chain_task: BaseLLMChain = Field(
        ..., description="LLM chain that perform the main task"
    )
    chain_reflection: BaseLLMChain = Field(
        ...,
        description="LLM chain that perform reflection on the result of the main task",
    )
    task_response_key: str = Field(
        default="context_response",
        description="The key to use for the task response in the message context.",
    )

    @field_validator("task_response_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context_"):
            raise ValueError("task_response_key must start with 'context_'")
        return v

    def execute(
        self, message: AgentMessage, keep_original_response: bool = True, **kwargs
    ) -> AgentMessage:
        self.state = "running"
        message = self.chain_task.invoke(message, **kwargs)
        message.execution_result = message.execution_result or "error"
        if message.execution_result != "success":
            message.error_message = (
                "Call chain not success. Received: " + message.execution_result
            )
            return message

        self.state = "reflecting"
        reflect_message = message.model_copy(deep=True)
        result_message = self.chain_reflection.invoke(reflect_message, **kwargs)
        self.state = "idle"
        result_message.origin = self.card.name
        if result_message.execution_result == "success" and keep_original_response:
            result_message.responses.insert(-1, message.responses[-1])

        return result_message


class LoopAgent(BaseAgent):
    """
    An agent that will loop until a certain condition is met.

    This agent takes two parameters: an agent to loop and a condition to stop the loop.
    The condition to stop the loop should be a function that takes an AgentMessage as an argument
    and returns a boolean indicating whether the loop should stop or not.

    Example:
        def is_stop(message: AgentMessage) -> bool:
            # stop when the message query is "stop"
            return message.query == "stop"

        agent = LoopAgent(
            agent=Agent1(),
            is_stop=is_stop
        )

        message = AgentMessage(query="hello")
        result = agent.execute(message)

    Attributes:
        agent: The agent to loop.
        is_stop: The condition to stop the loop.
    """

    agent: BaseAgent = Field(..., description="The agent to loop.")
    is_stop: Callable[[AgentMessage], bool] | Generator[bool] = Field(
        ...,
        description="The condition to stop the loop. It should be a function that takes an AgentMessage as an argument and returns a boolean.",
    )

    def execute(
        self,
        message: AgentMessage,
        keep_result: int | Callable[[Sequence[AgentResponse]], List[AgentResponse]] = 1,
        **kwargs,
    ) -> AgentMessage:
        self.state = "running"
        origin_len = len(message.responses)
        responses = message.model_copy(deep=True)

        def update_responses():
            if isinstance(keep_result, int) and keep_result > 0:
                responses.responses = responses.responses[-keep_result - origin_len :]
            elif isinstance(keep_result, Callable):
                responses.responses = keep_result(responses.responses)

        i = 0
        if isinstance(self.is_stop, Callable):
            while self.is_stop(message) is False:
                self.state = "running#" + str(i)
                i += 1
                message = self.agent.execute(message, **kwargs)
                if message.execution_result != "success":
                    break
                else:
                    responses.responses.append(message.responses.pop())
                update_responses()
        elif isinstance(self.is_stop, Iterator):
            try:
                while next(self.is_stop) is False:
                    self.state = "running#" + str(i)
                    i += 1
                    message = self.agent.execute(message, **kwargs)
                    if message.execution_result != "success":
                        break
                    else:
                        responses.responses.append(message.responses.pop())
                    update_responses()
            except StopIteration:
                pass
        else:
            raise TypeError(
                f"is_stop must be a callable or a Generator, received {type(self.is_stop)}"
            )

        self.state = "idle"
        responses.origin = self.card.name
        return responses

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, [self.agent], "sequential"
        )


class SequentialAgent(BaseAgent):
    """
    A sequential agent is an agent that will execute a list of agents in a particular order.

    The order of the agents is the order of the execution. This means that the first agent will be executed first, then the second agent and so on.

    Example:
        agent1 = Agent1()
        agent2 = Agent2()

        sequential_agent = SequentialAgent([agent1, agent2])

        message = AgentMessage(query="hello")
        result = sequential_agent.execute(message)

    Attributes:
        agents: The list of agents to execute in sequence.
    """

    agents: Sequence[BaseAgent] = Field(
        ...,
        description="""The list of agents to execute in sequence.
            The order of the sequence is the order of the execution.""",
        min_length=1,
    )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        for agent in self.agents:
            message = agent.execute(message, **kwargs)
            if message.execution_result != "success":
                break

        self.state = "idle"
        message.origin = self.card.name
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "sequential"
        )


class ParallelAgent(BaseAgent):
    """
    A parallel agent is an agent that will execute a list of agents in parallel.
    The agent will return a list of AgentMessage where each message is the result of the corresponding agent in the list.

    Example:
        agent1 = Agent1()
        agent2 = Agent2()

        parallel_agent = ParallelAgent([agent1, agent2])

        messages = [AgentMessage(query="hello"), AgentMessage(query="world")]
        results = parallel_agent.execute(messages)

    Attributes:
        agents: The list of agents to execute in parallel.
    """

    agents: Sequence[BaseAgent] = Field(
        ..., description="""The list of agents to execute in parallel""", min_length=1
    )

    def execute(  # type: ignore
        self, messages: AgentMessage | Sequence[AgentMessage], **kwargs
    ) -> AgentMessage:
        # If received a single message, all agents will process the same message
        if isinstance(messages, AgentMessage):
            messages = [messages.model_copy(deep=True)] * len(self.agents)
        elif len(messages) != len(self.agents):
            raise ValueError(
                "messages must be a list of AgentMessage with the same length as the number of agents"
            )

        self.state = "running"
        # if torch is used, we can use torch.multiprocessing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(agent.execute, message, **kwargs)
                for agent, message in zip(self.agents, messages)
            ]
            responses = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        result_message = AgentMessage(
            query="",
            origin=self.card.name,
            responses=[message.responses[-1] for message in responses],
            execution_result="success",
        )
        self.state = "idle"
        return result_message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "parallel"
        )


class CoordinatorAgent(BaseAgent):
    """
    This coordinator agent compose of 3 stages:
    - Planning: a planner agent will plan the steps to be executed.
    The planner recieves input query and returns a list of steps to be executed.
    For each step, it contains 3 fields:
        + worker: the worker agent that will execute the step
        + message: the instruction for the worker agent
        + dependencies: the dependencies of the step, this is picked from the result of the previous steps. The planner is responsible for choosing which results are needed for current step
    - Execution: For each step, an assigned worker will execute the step. In current version, steps are performed sequentially.
    - Summary: a summary agent will summarize the results so far and generate final answer.
    The summary agent recieves all results and write a final answer.
    This stage is optional.

    Example:
        planner_agent = PlannerAgent()
        worker_agent = WorkerAgent()
        summary_agent = SummaryAgent()

        coordinator_agent = CoordinatorAgent(planner_agent, worker_agent, summary_agent)

        message = AgentMessage(query="hello")
        result = coordinator_agent.execute(message)

    Attributes:
        planner_agent: The agent that will plan the steps to be executed.
        workers: The list of agents that will execute the steps.
        summary_agent: The agent that will summarize the results so far and generate final answer.
    """

    planner_agent: BaseAgent = Field(
        ..., description="""The agent that will plan the steps to be executed"""
    )
    parse_plan: Callable[
        [AgentMessage, Sequence[BaseAgent]],
        Sequence[Tuple[AgentMessage, BaseAgent, List]],
    ] = Field(
        ...,
        description="""The function that will parse the plan and return the list of steps to be executed.""",
    )
    workers: Sequence[BaseAgent] = Field(
        ...,
        description="""The list of agents that will execute the steps based on the plan""",
        min_length=1,
    )
    summary_chain: BaseLLMChain | None = Field(
        default=None,
        description="""The summary chain that will summarize the results so far and generate final answer""",
    )
    summary_prompt: str | None = Field(
        default=None,
        description="""The prompt that will be used by the summary chain""",
    )
    summary_steps_key: str = Field(
        default="context_results",
        description="""The key of which to store the context (e.g. intermediate result) during generation.
        Note that the key need to start with 'context_'""",
    )

    @field_validator("summary_steps_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context_"):
            raise ValueError("summary_steps_key must start with 'context_'")
        return v

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "planning"
        # TODO: abstract structured output
        message = self.planner_agent.execute(message, **kwargs)

        if message.execution_result != "success":
            return message

        steps = self.parse_plan(message, self.workers)

        # Validate parsed plan
        if not steps:
            self.state = "idle"
            message.execution_result = "error"
            message.error_message = (
                "Plan parsing failed: no steps returned from parse_plan"
            )
            return message

        if not isinstance(steps, Sequence):
            self.state = "idle"
            message.execution_result = "error"
            message.error_message = (
                f"Plan parsing failed: expected Sequence, got {type(steps)}"
            )
            return message

        for i, step in enumerate(steps):
            if not isinstance(step, tuple) or len(step) != 3:
                self.state = "idle"
                message.execution_result = "error"
                message.error_message = f"Plan parsing failed: step {i} is not a valid tuple of (message, agent, dependencies)"
                return message
            step_msg, step_agent, dependencies = step
            if not isinstance(step_msg, AgentMessage):
                self.state = "idle"
                message.execution_result = "error"
                message.error_message = (
                    f"Plan parsing failed: step {i} message is not an AgentMessage"
                )
                return message
            if not isinstance(step_agent, BaseAgent):
                self.state = "idle"
                message.execution_result = "error"
                message.error_message = (
                    f"Plan parsing failed: step {i} agent is not a BaseAgent"
                )
                return message
            if not isinstance(dependencies, list):
                self.state = "idle"
                message.execution_result = "error"
                message.error_message = (
                    f"Plan parsing failed: step {i} dependencies is not a list"
                )
                return message

        sub_task_result = []
        # TODO: check step dependency and parallelize steps to speed up
        summary_steps_key = self.summary_steps_key.replace("context_", "")
        for i, (step_msg, step_agent, dependencies) in enumerate(steps):
            step_msg.context = {summary_steps_key: sub_task_result}
            self.state = f"step {i}: worker {step_agent.card.name} running"
            sub_task_result.append(step_agent.execute(step_msg, **kwargs))

        if self.summary_chain is None:
            if self.summary_prompt is not None:
                # Use coordinator as summary agent
                self.state = f"{self.planner_agent.card.name} finalizing"
                summary_message = AgentMessage(
                    query=self.summary_prompt,
                    context={summary_steps_key: sub_task_result},
                    responses=[],
                )
                summary_message = self.planner_agent.execute(summary_message, **kwargs)
            else:
                return message
            # if no summary prompt is provided, just return the result from workers
        else:
            self.state = "finalizing"
            summary_message = AgentMessage(
                query=message.query,
                context={summary_steps_key: sub_task_result},
                responses=[],
            )
            summary_message = self.summary_chain.invoke(summary_message, **kwargs)

        self.state = "idle"
        summary_message.origin = self.card.name
        return summary_message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, [self.planner_agent, *self.workers], "sequential"
        )


class DebateAgent(BaseAgent):
    """
    Multiple agents participate in a discussion to solve a problem.
    The DebateAgent is designed to facilitate a discussion between multiple agents.
    It acts like a conversation manager who will decide the conversation flow

    Note that for the expandability, this agent is reponsible only for expanding the discussing topic, not the summary one.
    For the complete flow that performs topic rollout and finalize the solution, it is necessary to combine with other agent.
    There are several ways to achieve this: either by define a new agent that coodinate the actions or chain multiple agents together.
    """

    agents: Sequence[BaseAgent] = Field(
        ...,
        description="""The list of agents participate in the conversation""",
        min_length=1,
    )
    pick_strategy: (
        Literal["round_robin", "random", "simultaneous"]
        | Callable[[Sequence[BaseAgent]], BaseAgent]
    ) = Field(..., description="The strategy to pick the agent to run the next turn")
    random_seed: int | None = Field(
        default=None, description="The random seed for the 'random' pick strategy"
    )
    max_turns: int = Field(
        5,
        description="""The maximum number of debate turns to run.
        Note: for the simultaneous strategy, the actual number of turn will be factor of the number of agents""",
        ge=1,
    )
    should_stop: Callable[[AgentMessage], bool] | None = Field(
        default=None, description="The condition to stop the debate"
    )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        n = 0
        if self.pick_strategy == "random" and self.random_seed is not None:
            random.seed(self.random_seed)

        # All response turns are stored in the message.responses
        while n < self.max_turns:
            if self.pick_strategy == "round_robin":
                next_turn_agent = self.agents[n % len(self.agents)]
            elif self.pick_strategy == "random":
                next_turn_agent = random.choice(self.agents)
            elif isinstance(self.pick_strategy, Callable):
                next_turn_agent = self.pick_strategy(self.agents)
            elif self.pick_strategy == "simultaneous":
                self.state = f"turn {n}: all agents running"
                # TODO: parallelize the execution
                current_turn_messages = [
                    agent.execute(message.model_copy(deep=True), **kwargs)
                    for agent in self.agents
                ]
                message.responses.extend(
                    [msg.responses[-1] for msg in current_turn_messages]
                )
                n += len(self.agents)
                message.execution_result = "success"
            else:
                raise ValueError(
                    "pick_strategy must be one of 'round_robin', 'random', 'simutaneous' or a function"
                )

            if self.pick_strategy != "simultaneous":
                self.state = f"turn {n}: agent {next_turn_agent.card.name} running"  # type: ignore
                message = next_turn_agent.execute(message, **kwargs)  # type: ignore
                if len(message.responses) > 0:
                    pass
                n += 1

            if message.execution_result != "success" or (
                self.should_stop and self.should_stop(message)
            ):
                break

        self.state = "idle"
        message.origin = self.card.name
        message.execution_result = "success"
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self,
            self.agents,
            "parallel" if self.pick_strategy == "simultaneous" else "sequential",
        )


class VotingAgent(BaseAgent):
    """
    A voting agent selects the best answer among a list of candidates responses.
    """

    agents: Sequence[BaseAgent] = Field(
        ...,
        description="""The list of agents participate in the voting""",
        min_length=1,
    )
    voting_method: Literal["majority_vote", "llm_score"] = Field(
        ...,
        description="""The voting method to use.
        1. majority_vote: It uses bleu or rouge score to calculate score for candidates
        2. llm_score: use LLM to score the candidate messages.
            For each message, it will score messages generated by other agents.
            Then score for each message will be aggregated (average) to get the final score.

        Note:
        - The majority_vote method can also be known as the metric-based scorer.
            i.e. Use a metric to score the candidate with reference message. This approach is simple and fast.
        - The llm_score method can also be known as the model-based scorer.
            i.e. Use a model to score the candidate message.
            This approach is more complex and slower. But it can capture more scenarios like semantic similarity, context similarity, etc.
        """,
    )
    scorer: str | Callable[[str], float] = Field(
        ...,
        description="""1. If voting_method is 'llm_score', the function to use to convert LLM response to score.
        Note that LLM may generate reponse that could not convert to score.
        In these cases, there are various ways to handle.
        For example, we just ignore it and assign the score to 0

        2. If voting_method is 'majority_vote'. scorer accept bleu or rouge score
            - 'bleu' or 'agent_forest': Use bleu score to calculate voting score. https://arxiv.org/pdf/2402.05120.
            - 'rougeL': use rouge-L (rouge longest common sequence)
            - 'rougeN': N can be an integer, e.g. rouge1, rouge2,...""",
    )
    voting_prompt: str | None = Field(
        default=None,
        description="The prompt to use for the voting. This parameter is used when voting_method is 'llm_score'",
    )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        if self.voting_method == "majority_vote":
            # sanity check
            if not isinstance(self.scorer, str):
                raise TypeError(
                    "When voting_method is 'majority_vote'. scorer need to be one of bleu, agent_forest, rougeL or rouge[N]"
                )
            elif self.scorer == "agent_forest" or self.scorer == "bleu":
                # Use the same library and code snippet as the author
                # Source: https://github.com/MoreAgentsIsAllYouNeed/AgentForest/blob/master/src/utils.py#L220
                def bleu_score(hyp: str, ref: str) -> float:
                    return sentence_bleu(hyp, [ref], lowercase=True).score

                score_func = bleu_score
            elif re.match(r"^rouge(L|\d+)$", self.scorer):
                scorer = rouge_scorer.RougeScorer([self.scorer], use_stemmer=True)

                def rouge_score(hyp: str, ref: str) -> float:
                    return list(scorer.score(ref, hyp).values())[0].fmeasure

                score_func = rouge_score
            else:
                raise ValueError("Scorer not supported")

            scores = {agent.card.name: 0.0 for agent in self.agents}
            message_map = {name: msg for name, msg in message.responses}
            for agent_name, msg in message.responses:
                total_score = 0.0
                for other_agent in self.agents:
                    if agent_name == other_agent.card.name:
                        continue
                    total_score += score_func(message_map[other_agent.card.name], msg)
                scores[agent_name] = total_score
            highest_score_agent = max(scores, key=scores.get)  # type: ignore
            message.responses.append((self.card.name, message_map[highest_score_agent]))

        elif self.voting_method == "llm_score":
            if self.voting_prompt is None:
                raise ValueError(
                    "voting_prompt is required for llm_score voting method"
                )
            elif not isinstance(self.scorer, Callable):
                raise TypeError(
                    "scorer must be a Callable when voting_method is 'llm_score'"
                )

            scores = {agent.card.name: 0.0 for agent in self.agents}
            message_map = {name: msg for name, msg in message.responses}
            for response in message.responses:
                total_score = 0
                for agent in self.agents:
                    if response[0] == agent.card.name:
                        continue

                    try:
                        voting_message = AgentMessage(
                            query=self.voting_prompt, responses=[]
                        )
                        voting_message = agent.execute(voting_message, **kwargs)
                        total_score += self.scorer(voting_message.responses[-1][1])
                    except Exception:
                        # unable to parse the score
                        pass
                scores[response[0]] = total_score
                # The highest or average score will return the same response
                # Additional note: because of the score, we even can rank all candidate responses and output a leaderboard.
            highest_score_agent = max(scores, key=scores.get)  # type: ignore
            message.responses.append((self.card.name, message_map[highest_score_agent]))

        else:
            raise ValueError(
                "voting_method must be one of 'llm_score', 'majority_vote'"
            )

        self.state = "idle"
        message.origin = self.card.name
        message.execution_result = "success"
        return message

    def _set_composed_state(self) -> None:
        # TODO: find way to parallel voting of all agents
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "sequential"
        )
