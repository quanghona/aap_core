import abc
import re
import warnings
from collections.abc import Callable, Sequence
from importlib import resources as importlib_resources
from typing import Any, Dict, List, Literal, Tuple

import gymnasium as gym
import nltk
import numpy as np
import torch
from gymnasium import spaces
from pydantic import Field, PrivateAttr, field_validator
from rbloom import Bloom
from rensa import CMinHash, CMinHashDeduplicator, RMinHash, RMinHashLSH  # type: ignore

from aap_core.dedup import (
    BloomAlgoConfig,
    LSHBloom,
    LSHBloomAlgoConfig,
    MinHashAlgoConfig,
    MinHashLSHAlgoConfig,
    SimHash,
    SimHashAlgoConfig,
    SuffixAlgoConfig,
    SuffixArray,
)

from .policy import BasePolicy

# import toon_format
from .types import AgentMessage, BaseChain, BaseLLMChain


class BasePromptAugmenter(BaseChain):
    """A base class to enhance / rewrite the prompt.

    There are two types of prompt augmenter:
    - Data augmenter: Give more context to the prompt by adding external data
        + files (CSV, JSON, Markdown, etc.)
        + database (SQL,...)
        + RAG
        + web search
    Common alternative is Retriever
    - Structure augmenter: Rewrite / refine the prompt partially or entirely.
    """

    loop: int | Callable[[AgentMessage], bool] | None = Field(
        default=None,
        description="The loop, either by number of times or by stop condition",
    )

    async def acall(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    @abc.abstractmethod
    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    def call(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.augment(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if isinstance(self.loop, int):
            for _ in range(self.loop):
                message = self.call(message, **kwargs)
            return message
        elif callable(self.loop):
            while self.loop(message):
                message = self.call(message, **kwargs)
            return message
        else:
            return self.call(message, **kwargs)


class IdentityPromptAugmenter(BasePromptAugmenter):
    """A prompt enhancer that does nothing.
    This serves as a default prompt enhancer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message


class SimplePromptAugmenter(BasePromptAugmenter):
    """A prompt enhancer that uses a template to augment the prompt.
    Specifically, it simply concatenate the original prompt with the context data,
    using provided format.

    This augmenter mainly used for simple scenarios such as tabular data concatenation or naive RAG.
    In other word, this naturally suitables for data augmenter type.
    """

    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    data_key: str = Field(
        default="context.data", description="The key to the context data in the message"
    )

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{data}" not in v:
            raise ValueError("The format must contain at least {query} and {data}")
        return v

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("data_key must start with 'context.'")
        return v

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if message.context is None:
            raise ValueError("Message context is None")
        elif self.data_key not in message.context:
            raise ValueError(f"Message context does not contain {self.data_key}")
        context_key = self.data_key.replace("context.", "")
        message.query = self.format.format(
            query=message.query, data=message.context[context_key], **kwargs
        )
        return message


class MetaPromptAugmenter(BasePromptAugmenter):
    """A simple prompt augmenter that use an LLM chain to rewrite the prompt."""

    chain: BaseLLMChain = Field(..., description="LLM chain that rewrite the prompt")

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.chain.invoke(message, **kwargs)
        return message


class DeduplicationPromptAugmenter(BasePromptAugmenter):
    """A prompt augmenter that deduplicates the generated prompt.

    A prompt with repeated sentences makes the LLM model harder to focus on the
    correct information and wastes token budget. This class focuses on exact or
    near-duplications at the sentence level.

    Supports following algorithms:
    - **minhash**: Uses C-MinHash with CMinHashDeduplicator for exact deduplication
    - **minhash_lsh**: Uses R-MinHash with LSH (Locality-Sensitive Hashing) for
      efficient near-duplicate detection at scale
    - **bloomfilter**: Uses Bloom Filter for fast probabilistic exact deduplication
    - **simhash**: Uses custom numpy-based SimHash for near-duplicate detection
    - **lsh_bloom**: Uses MinHash with LSH-Bloom Filter for space-efficient
      near-duplicate detection at very large scale
    - **suffix_array**: Uses Suffix Array with LCP array for exact substring
      deduplication. Inspired by Google's deduplicate-text-datasets and
      Chenghao Mou's text-dedup.

    We do not use semantic deduplication. Semantic similarity, in usual cases,
    helps strengthen the context in the prompt and helps the model more likely
    to generate the right answer.
    """

    _dedup: Any = PrivateAttr()
    _algo_config: Any = PrivateAttr()
    _dedup_func: Callable[[List[str]], List[str]] = PrivateAttr()

    def __init__(
        self,
        algo_args: Dict[str, Any],
        **kwargs,
    ):
        # Validate and normalize algorithm_name
        algo_name = algo_args.get("algorithm_name", "minhash")
        if algo_name == "minhash":
            algo_config = MinHashAlgoConfig(**algo_args)
            dedup = CMinHashDeduplicator(
                threshold=algo_config.threshold,
                num_perm=algo_config.num_perm,
                seed=algo_config.seed,
            )
            dedup_func = self._deduplicate_minhash
        elif algo_name == "minhash_lsh":
            algo_config = MinHashLSHAlgoConfig(**algo_args)
            dedup = RMinHashLSH(
                threshold=algo_config.threshold,
                num_perm=algo_config.num_perm,
                num_bands=algo_config.num_bands,
            )
            dedup_func = self._deduplicate_minhash_lsh
        elif algo_name == "bloomfilter":
            algo_config = BloomAlgoConfig(**algo_args)
            dedup = Bloom(
                expected_items=algo_config.expected_items,
                false_positive_rate=algo_config.false_positive_rate,
            )
            dedup_func = self._deduplicate_bloomfilter
        elif algo_name == "simhash":
            algo_config = SimHashAlgoConfig(**algo_args)
            dedup = SimHash(
                hash_bits=algo_config.hash_bits,
            )
            dedup_func = self._deduplicate_simhash
        elif algo_name == "lsh_bloom":
            algo_config = LSHBloomAlgoConfig(**algo_args)
            dedup = LSHBloom(
                threshold=algo_config.threshold,
                num_perm=algo_config.num_perm,
                num_bands=algo_config.num_bands,
                expected_items=algo_config.expected_items,
                false_positive_rate=algo_config.false_positive_rate,
            )
            dedup_func = self._deduplicate_lsh_bloom
        elif algo_name == "suffix_array":
            algo_config = SuffixAlgoConfig(**algo_args)
            dedup = None  # Suffix array builds on-demand in _deduplicate_suffix_array
            dedup_func = self._deduplicate_suffix_array
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        super().__init__(**kwargs)

        # Set private attributes AFTER super().__init__() (Pydantic v2 requirement)
        self._algo_config = algo_config  # type: ignore[attr-defined]
        self._dedup = dedup  # type: ignore[attr-defined]
        self._dedup_func = dedup_func  # type: ignore[attr-defined]

    def _deduplicate_minhash(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using C-MinHash with CMinHashDeduplicator.

        Uses exact minhash comparison via is_duplicate() and add() methods.
        Clears the deduplication index once before processing all sentences.
        """
        self._dedup.clear()
        deduped_sentences = []
        for sent in sentences:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            if not tokens:
                deduped_sentences.append(sent)
                continue
            mh = CMinHash(
                num_perm=self._algo_config.num_perm,
                seed=self._algo_config.seed,
            )
            mh.update(tokens)
            if not self._dedup.is_duplicate(f"sent_{len(deduped_sentences)}", mh):
                self._dedup.add(f"sent_{len(deduped_sentences)}", mh)
                deduped_sentences.append(sent)
        return deduped_sentences

    def _deduplicate_minhash_lsh(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using R-MinHash with LSH.

        Creates a new RMinHashLSH instance per call (no clear() method).
        Uses query() to check for similar items, then insert() to add non-duplicates.
        """
        self._dedup = RMinHashLSH(
            threshold=self._algo_config.threshold,
            num_perm=self._algo_config.num_perm,
            num_bands=self._algo_config.num_bands,
        )
        deduped_sentences = []
        for sent in sentences:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            if not tokens:
                deduped_sentences.append(sent)
                continue
            mh = RMinHash(
                num_perm=self._algo_config.num_perm,
                seed=self._algo_config.seed,
            )
            mh.update(tokens)
            # query() returns a list of keys of potentially similar items
            candidates = self._dedup.query(mh)
            if candidates:
                # Found similar items in the index, skip this sentence
                continue
            # Not a duplicate, insert it
            self._dedup.insert(len(deduped_sentences), mh)
            deduped_sentences.append(sent)
        return deduped_sentences

    def _deduplicate_bloomfilter(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using Bloom Filter.

        Uses rbloom Bloom Filter for exact deduplication based on sentence hashing.
        Clears the bloom filter once before processing all sentences.
        """
        self._dedup.clear()
        deduped_sentences = []
        for sent in sentences:
            if sent in self._dedup:
                # Sentence already in bloom filter, skip it
                continue
            # Not a duplicate, add it
            self._dedup.add(sent)
            deduped_sentences.append(sent)
        return deduped_sentences

    def _deduplicate_simhash(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using SimHash.

        Computes a SimHash fingerprint for each sentence and compares fingerprints
        using Hamming distance. Sentences with similarity >= threshold are considered
        duplicates and removed.
        """
        self._dedup = SimHash(
            hash_bits=self._algo_config.hash_bits,
        )
        deduped_sentences = []
        fingerprints = []
        threshold = self._algo_config.threshold

        for sent in sentences:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            if not tokens:
                deduped_sentences.append(sent)
                fingerprints.append(None)
                continue

            fp = self._dedup.compute_hash(sent)
            is_duplicate = False

            # Compare with all existing fingerprints
            for existing_fp in fingerprints:
                if existing_fp is None:
                    continue
                sim = SimHash.similarity(fp, existing_fp)
                if sim >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped_sentences.append(sent)
                fingerprints.append(fp)

        return deduped_sentences

    def _deduplicate_lsh_bloom(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using LSH-Bloom Filter.

        Uses MinHash with LSH-Bloom Filter for space-efficient near-duplicate detection.
        Creates a new LSHBloom instance per call (no clear() method needed as it's
        recreated for each deduplication pass).

        Inspired by the LSHBloom algorithm from https://arxiv.org/abs/2411.04257
        """
        self._dedup = LSHBloom(
            threshold=self._algo_config.threshold,
            num_perm=self._algo_config.num_perm,
            num_bands=self._algo_config.num_bands,
            expected_items=self._algo_config.expected_items,
            false_positive_rate=self._algo_config.false_positive_rate,
        )
        deduped_sentences = []
        for sent in sentences:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            if not tokens:
                deduped_sentences.append(sent)
                continue
            mh = RMinHash(
                num_perm=self._algo_config.num_perm,
                seed=self._algo_config.seed,
            )
            mh.update(tokens)
            # Query first to check for potential duplicates
            if self._dedup.query(mh):
                # Potential duplicate found, skip this sentence
                continue
            # Not a duplicate, insert it
            self._dedup.insert(f"sent_{len(deduped_sentences)}", mh)
            deduped_sentences.append(sent)
        return deduped_sentences

    def _deduplicate_suffix_array(self, sentences: List[str]) -> List[str]:
        """Deduplicate sentences using Suffix Array with LCP array.

        This method works on character level to find exact duplicate substrings
        >= min_length. Sentences containing duplicate characters are removed.

        Inspired by Google's deduplicate-text-datasets and Chenghao Mou's text-dedup.
        """
        # Create suffix array instance on-demand
        suffix_array = SuffixArray(
            min_length=self._algo_config.min_length,
            max_length=self._algo_config.max_length,
        )

        # find_duplicates handles: joining sentences, lowercasing, ord conversion, SA build, duplicate finding
        # Returns bytearray where duplicate_positions[i] == 1 means position i is a duplicate
        duplicate_positions = suffix_array.find_duplicates(sentences)

        if not duplicate_positions or all(b == 0 for b in duplicate_positions):
            return sentences

        # Map duplicate character positions back to sentences
        # Build a mapping from character position to sentence index
        sentence_boundaries = []
        current_pos = 0
        for sent in sentences:
            sentence_boundaries.append((current_pos, current_pos + len(sent)))
            current_pos += len(sent) + 1  # +1 for space separator

        # Find sentences where the ENTIRE sentence is a duplicate
        # Group by content and keep only the first occurrence
        # Use bytearray for memory efficiency: 1 byte per sentence vs. ~28 bytes per int in set
        num_sentences = len(sentences)
        sentences_to_remove = bytearray(num_sentences)
        seen_content = {}
        for sent_idx, (start, end) in enumerate(sentence_boundaries):
            # Check if all characters in this sentence are in duplicate positions
            # bytearray index access is O(1) and faster than set membership
            if all(duplicate_positions[pos] == 1 for pos in range(start, end)):
                # This sentence is entirely in duplicate regions
                # Get its normalized content for grouping
                sent_content = sentences[sent_idx].strip().lower()
                if sent_content in seen_content:
                    # Subsequent duplicate, mark for removal
                    sentences_to_remove[sent_idx] = 1
                else:
                    # First occurrence, keep it
                    seen_content[sent_content] = sent_idx

        # Return non-duplicate sentences
        deduped_sentences = [
            sent for idx, sent in enumerate(sentences) if not sentences_to_remove[idx]
        ]
        return deduped_sentences

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if not message.query:
            return message

        sentences = nltk.sent_tokenize(message.query.strip())
        deduped_sentences = self._dedup_func(sentences)
        message.query = " ".join(deduped_sentences)
        return message


SEEDataSet = Sequence[Tuple[str, str]]  # A dataset is a sequence of input-output pairs
SEEPerformanceTuple = Tuple[
    Sequence[float], float
]  # A performance tuple contains a performance vector and a performance score


class SEEPromptAugmenter(BasePromptAugmenter):
    """# SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization
    https://arxiv.org/abs/2402.11347

    This is a self implementation that adapt to ours aap framework with additional extensions.
    The logic may not be identical to the original work.

    SEE uses LLM operators to perform generation and variation. There are 5 operators introduced in the work:
    - Lamarckian: reverse engineering by generating prompt from a set of input-output pairs
    - EDA (Estimation of Distribution): takes in a group of candidates and outputs a new candidate by studying the input group.
    - Crossover: mixing the traits of both parents and generates a new candidate
    - Feedback: use 2 agents - Examiner and Improver to generate new candidate
    - Semantic: modifies the candidate lexically while preserving its semantic meaning

    The SEE framework contains 4 phases:
    - Phase 0: Global initialization
    - Phase 1: Local feedback operation
    - Phase 2: Global fusion operation
    - Phase 3: Local semantic operation

    Regarding prompts for operators, we have 3 ways to distribute prompt into user prompt and system prompt:
    - Put all prompt content in the system prompt template and user prompt is empty
    - The system prompt is the fixed instruction, and the template and examples are in the user prompt. This is the recommended implementation
    - The system prompt is empty and all the prompt content stay in the user prompt. This approach has highest flexibility but only suitable for development

    To maximize the effect of operators, we should adjust the temperature of LLM accordingly.
    Push the temperature higher for exploring phases (0 and 2), 3 global operators: Lamarckian, EDA, Crossover.
    Lower the temperature for exploiting phases (1 and 3), 2 local operators: Feedback, Semantic.
    The adjustment stays inside the logic of the chains, which is outside of this SEE class

    Note that we could use the same or different LLMs for different operators and the baseline.

    As stated in the paper, SEE needs multiple iterations and relatively large amount of API call, which might be inefficient for large-scale production.
    So this is only for experimental use.

    Following our framework, in the prompt template, the keyword need to be prefix with 'context.' and the additional
    data need to store in 'context' field of the AgentMessage object

    The custom scorer signature when initalization is Callable[[BaseLLMChain, str, Sequence[str], Sequence[SEEPerformanceTuple], SEEDataSet, ...], SEEPerformanceTuple | None]
    """

    _scorer: Callable[..., SEEPerformanceTuple | None] = PrivateAttr()
    _dist_func: Callable[[Sequence[float], Sequence[float]], float] = PrivateAttr()
    _scorer_args: Dict = PrivateAttr()
    _eval_method: Callable[[str, str], bool] = PrivateAttr()

    base_chain: BaseLLMChain = Field(
        ..., description="The base LLM chain used by SEE to generate result"
    )
    lamarckian_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Lamarckian operator"
    )
    eda_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for EDA operator"
    )
    crossover_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Crossover operator"
    )
    examiner_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Feedback operator"
    )
    improver_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Feedback operator"
    )
    semantic_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Semantic operator"
    )

    dev_set: SEEDataSet = Field(
        ...,
        description="The dataset for evaluate prompt. This is D_dev in the algorithm",
        min_length=1,
    )
    init_data: SEEDataSet | str = Field(
        ...,
        description="""The initial data for phase 0. There are two types of initialzation:
        - See-io-pair: provide a set of input-output pairs. SEE apply Lamarckian to generate prompts
        - SEE-example: SEE take a initial prompt and use Semantic to generate new prompts
        """,
    )

    lamarckian_message: AgentMessage | None = Field(
        default=None,
        description="""The prompt use for Lamarckian operator.
        If not provided, the default prompt template will be constructed,
        The default key for pairs is {context.pairs}.
        """,
    )
    eda_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for EDA operator.
    If not provided, the default prompt template will be constructed,
    The default key for candidates is {context.candidates}.
    """,
    )
    crossover_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for Crossover operator.
    If not provided, the default prompt template will be constructed,
    The default key for parents is {context.parents}.
""",
    )
    examiner_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for examine candidate, for feedback operator
    If not provided, the default prompt template will be constructed
    the default key for wrong cases is {context.wrong_cases}
""",
    )
    improver_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use to improve the existing candidate, for feedback operator.
    If not provided, the default prompt template will be constructed,
    The default key for feedback is {context.feedback}
""",
    )
    semantic_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for Semantic operator.
    If not provided, the default prompt template will be constructed,
    The default key for candidate is {context.candidate}.
    """,
    )

    pool_size_0: int = Field(
        default=15,
        description="Pool size of phase 0: Global initialization. Marked as n_0 in the algorithm",
    )
    pool_size_1: int = Field(
        default=5,
        description="Pool size of phase 1: Local feedback. Marked as n_1 in the algorithm",
    )
    pool_size_2: int = Field(
        default=5,
        description="Pool size of phase 2: Global fusion. Marked as n_2 in the algorithm",
    )
    pool_size_3: int = Field(
        default=5,
        description="Pool size of phase 3: Local semantic. Marked as n_3 in the algorithm",
    )
    tolerance_1: int = Field(
        default=1,
        description="Tolerance for phase 1: Local feedback, aka for feedback operator. This is marked as K_1 in the algorithm",
    )
    tolerance_2: int = Field(
        default=8,
        description="""Tolerance for phase 2: Global fusion, aka for EDA and crossover operator. This is marked as K_2 in the algorithm.
        Note that this is tolerance for whose phase, which is total tolerance for 2 operators""",
    )
    tolerance_3: int = Field(
        default=1,
        description="Tolerance for phase 3: Local semantic, aka for semantic operator. This is marked as K_3 in the algorithm",
    )
    performance_gain_threshold: float = Field(
        default=0.01, description="Performance gain threshold. In percent scale"
    )
    num_crossover_parents: int = Field(
        default=2,
        description="""Number of parents to use for exploring operator such as EDA and Crossover.
        Setting value to -1 will use all candidates in the prompt pool.""",
    )
    num_eda_parents: int = Field(
        default=-1,
        description="""Number of parents to use for exploring operator such as EDA and Crossover
        Setting value to -1 will use all candidates in the prompt pool.""",
    )
    eda_with_index: bool = Field(
        default=False,
        description="Whether to rank candidates",
    )
    crossover_with_distinct: bool = Field(
        default=False,
        description="Whether to consider the diversity in parents",
    )
    num_feedback_wrongcases: int = Field(
        default=3,
        description="Number of wrong cases to include in the prompt for feedback operator",
        gt=0,
    )
    eda_parent_selection: Literal["wheel", "random", "tournament"] = Field(
        default="random",
        description="Parent selection strategy for EDA operator",
    )
    crossover_parent_selection: Literal["wheel", "random", "tournament"] = Field(
        default="random",
        description="""Parent selection strategy for crossover operator.
        This strategy applies when crossover_with_distinct is True.
        The method will select 1 parent initially and the rest is selected by maximizing the total distance between parents
        """,
    )

    def __init__(
        self,
        scorer: Literal["hamming"]
        | Callable[..., SEEPerformanceTuple | None] = "hamming",
        dist_func: Callable[[Sequence[float], Sequence[float]], float] | None = None,
        scorer_args: Dict = {},
        eval_method: Literal["exact", "include"] | Callable[[str, str], bool] = "exact",
        **kwargs,
    ):
        warnings.warn(
            "This class is only for experimental use. It is not recommended for production use."
        )

        super().__init__(**kwargs)
        if scorer == "hamming":
            self._scorer = self._hamming_scorer
            self._dist_func = SEEPromptAugmenter._hamming_distance
        # elif scorer == "levenshtein":
        #     self._scorer = SEEPromptAugmenter._levenshtein_scorer
        # elif scorer == "cosine":
        #     self._scorer = SEEPromptAugmenter._cosine_scorer
        elif isinstance(scorer, Callable) and dist_func is not None:
            self._scorer = scorer
            self._dist_func = dist_func
        else:
            raise ValueError("scorer not supported")
        self._scorer_args = scorer_args

        if eval_method == "exact":
            self._eval_method = lambda x, y: x == y
        elif eval_method == "include":
            self._eval_method = lambda x, y: x in y
        elif isinstance(eval_method, Callable):
            # various methods can be tried like bleuscore, rouge, BERTscore, CTC, LLM-as-a-judge, etc
            self._eval_method = eval_method
        else:
            raise ValueError("eval_method not supported")

    def score(
        self,
        prompt: str,
        dataset: SEEDataSet,
    ) -> SEEPerformanceTuple:
        performance_vector = []
        for data in dataset:
            query = f"""{prompt}
            Question:
            {data[0]}
            Answer:"""
            message = self.base_chain.invoke(AgentMessage(query=query))
            performance_vector.append(
                int(self._eval_method(data[1], message.responses[-1][1]))
            )
        return performance_vector, sum(performance_vector) / len(performance_vector)

    @classmethod
    def _hamming_distance(cls, v1: Sequence[float], v2: Sequence[float]) -> int:
        return int(np.count_nonzero(np.array(v1) != np.array(v2)))

    def _hamming_scorer(
        self,
        chain: BaseLLMChain,
        prompt: str,
        pool: Sequence[str],
        performance_pool: Sequence[SEEPerformanceTuple],
        dataset: SEEDataSet,
        distance_threshold: int = 2,
    ) -> SEEPerformanceTuple | None:
        perf_vec, score = self.score(prompt, dataset)
        # If pool is empty (first iteration), accept the candidate
        if len(performance_pool) == 0:
            return (perf_vec, score)
        # can only check with lowest score candidate
        min_dist = min(
            SEEPromptAugmenter._hamming_distance(perf_vec, p[0])
            for p in performance_pool
        )  # hamming distance
        # Found similar candidate -> NOT use this prompt
        return None if min_dist < distance_threshold else (perf_vec, score)

    @classmethod
    def _sort_pool(
        cls, P_t: List[str], S_t: List[SEEPerformanceTuple], pool_size: int
    ) -> Tuple[List[str], List[SEEPerformanceTuple]]:
        # S_t elements are SEEPerformanceTuple = (performance_vector, score)
        # Sort by score (index 1) in descending order
        sorted_indices = np.argsort([s[1] for s in S_t])[::-1]
        P_t = [P_t[i] for i in sorted_indices]
        S_t = [S_t[i] for i in sorted_indices]
        return P_t[:pool_size], S_t[:pool_size]

    @classmethod
    def _load_default_prompt(cls, filename: str) -> str:
        """Load a default prompt file from the package resources."""
        with (
            importlib_resources.files("aap_core.default_prompts")
            .joinpath(filename)
            .open("r") as f
        ):
            return f.read()

    @classmethod
    def _selection_random(
        cls, num_parents: int, S: Sequence[SEEPerformanceTuple]
    ) -> List[int]:
        return list(np.random.choice(len(S), num_parents, replace=False))

    @classmethod
    def _selection_wheel(
        cls, num_parents: int, S: Sequence[SEEPerformanceTuple]
    ) -> List[int]:
        total_score = sum([s[1] for s in S])
        probabilities = [s[1] / total_score for s in S]
        parents = []
        while len(parents) < num_parents:
            cumulative_probabilities = [
                sum(probabilities[: i + 1]) for i in range(len(probabilities))
            ]
            random_value = np.random.uniform(0, 1)
            parent_index = next(
                i
                for i, value in enumerate(cumulative_probabilities)
                if value >= random_value
            )
            if parent_index not in parents:
                parents.append(parent_index)
        return parents

    @classmethod
    def _selection_tournament(
        cls, num_parents: int, S: Sequence[SEEPerformanceTuple]
    ) -> List[int]:
        parents = []
        while len(parents) < num_parents:
            parent_indices = np.random.choice(len(S), 2)
            parent_index = np.argmax([S[i][1] for i in parent_indices])
            selected = parent_indices[parent_index]
            if selected not in parents:
                parents.append(selected)
        return parents

    # TODO: extend algorithm to multimodal data
    def lamarckian(self, pairs: SEEDataSet, **kwargs) -> str:
        """
        Generate new prompt using Lamarckian operator
        This method works with corresponding lamarckian_message and lamarckian_chain attributes.
        In the prompt in lamarckian_message, it must contains key 'context.pairs' to fetch data from provided dataset.

        Args:
            pairs (SEEDataSet): input-output pairs of the data
            **kwargs: arguments for lamarckian chain

        Returns:
            str: the generated prompt
        """
        dataset = "\n\n".join(
            [f"Input: {pair[0]}\nOutput: {pair[1]}" for pair in pairs]
        )
        if self.lamarckian_message is None:
            # The default prompt in the paper
            default_lamarckian_prompt = SEEPromptAugmenter._load_default_prompt(
                "see_default_lamarckian.md"
            )
            message = AgentMessage(
                query=default_lamarckian_prompt,
                context={"pairs": dataset},
            )
        else:
            message = self.lamarckian_message
            if message.context is None:
                message.context = {"pairs": dataset}
            else:
                message.context["pairs"] = dataset
        msg = self.lamarckian_chain.invoke(message, **kwargs)
        return msg.responses[-1][1]

    def eda(
        self, candidates: Sequence[str], S: Sequence[SEEPerformanceTuple], **kwargs
    ) -> str:
        """
        Generate new prompt using EDA operator
        This method works with corresponding eda_message and eda_chain attributes.

        In the prompt in eda_message, it must contains key 'context.candidates' to fetch candidates.
        We can control number of candidates by setting num_eda_parents.
        If we not apply indexing, the candiadates will be shuffled before fetch into prompt message

        Args:
            candidates (Sequence[str]): list of candidate prompts
            **kwargs: arguments for eda chain

        Returns:
            str: the generated prompt
        """
        num_parents = (
            len(candidates)
            if self.num_eda_parents < 0
            else min(len(candidates), self.num_eda_parents)
        )
        selection_dict = {
            "random": SEEPromptAugmenter._selection_random,
            "tournament": SEEPromptAugmenter._selection_tournament,
            "wheel": SEEPromptAugmenter._selection_wheel,
        }
        indices = selection_dict[self.eda_parent_selection](num_parents, S)
        if self.eda_with_index:
            indices = np.sort(indices)
        parents = [candidates[i] for i in list(map(int, indices))]
        cand_str = "\n\n".join(parents)
        if self.eda_message is None:
            default_eda_prompt = SEEPromptAugmenter._load_default_prompt(
                "see_default_eda.md"
            )
            message = AgentMessage(
                query=default_eda_prompt,
                context={"candidates": cand_str},
            )
        else:
            message = self.eda_message
            if message.context is None:
                message.context = {"candidates": cand_str}
            else:
                message.context["candidates"] = cand_str
        message = self.eda_chain(message, **kwargs)
        return message.responses[-1][1]

    @classmethod
    def max_vector_distance_subarray(
        cls,
        S: Sequence[SEEPerformanceTuple],
        k: int,
        dist_func: Callable[[Sequence[float], Sequence[float]], float],
        selection_method: Literal["tournament", "wheel", "random"] = "random",
    ) -> Sequence[int]:
        if k > len(S):
            raise ValueError("k cannot be greater than the length of S.")
        if k < 2:
            return list(range(k))

        selection_dict = {
            "random": SEEPromptAugmenter._selection_random,
            "tournament": SEEPromptAugmenter._selection_tournament,
            "wheel": SEEPromptAugmenter._selection_wheel,
        }
        selected_indices = selection_dict[selection_method](1, S)

        # Greedily add points until we reach size k
        while len(selected_indices) < k:
            best_next_idx = -1
            max_total_extra_dist = -1

            for i in range(len(S)):
                if i in selected_indices:
                    continue

                # Calculate how much this point adds to the total pairwise distance
                current_extra_dist = sum(
                    dist_func(S[i][0], S[idx][0]) for idx in selected_indices
                )

                if current_extra_dist > max_total_extra_dist:
                    max_total_extra_dist = current_extra_dist
                    best_next_idx = i

            selected_indices.append(best_next_idx)

        return selected_indices

    def crossover(
        self, P: Sequence[str], S: Sequence[SEEPerformanceTuple], **kwargs
    ) -> str:
        """
        Generate new prompt using Crossover mutation
        This method works with corresponding crossover_message and crossover_chain attributes.

        In the prompt in crossover_message, it must contains key 'context.parents' to fetch parents.
        We can control number of parents by setting num_crossover_parents.

        In the original work, the operator only accepts 2 parents. We generalize it
        to accept any number of parents. If diversity is not considered, we choose the best k parents.
        Otherwise we formulate the problem as below:

        Given a set of n parents P with its corresponding performance S, a target number of selected parents k.
        Find a subset of k parents that maximize the total pairwise distance between performance vector of selected parents.

        Hence, the definition 4 (Crossover Operator - CR) in the paper is generalized:
        > Crossover generates a new candidate based on k parents. It is a function
        > operator O_C that perform O_C(P_k, L) = p' where P_k is a subset of P,
        > p' is the generated prompt that hopefully hold features from prompts in P_k.
        > If P_k is chosen from P so that it maximizes \\sum_{i < j}^{P_k}d(p_i, p_j), we call it Crossover + Distinct (CR+D)

        If k = 2, it fallbacks to the original work
        We gather P_k by greedily adding points until we reach size k. First we
        find 2 initial points that have the largest distance among all pairs.
        Then for the rest points, we iteratively find the point that have max distance
        with the current set. The implementation of the algorithm is in `max_vector_distance_subarray` method

        Selected parents will be shuffled before fetch into prompt message

        Args:
            P (Sequence[str]): input prompts
            S (Sequence[SEEPerformanceTuple]): corresponding performance vectors.
            **kwargs: arguments for crossover chain

        Returns:
            str: the generated prompt
        """
        num_parents = (
            len(P)
            if self.num_crossover_parents < 0
            else min(len(P), self.num_crossover_parents)
        )

        if self.crossover_with_distinct:
            indices = SEEPromptAugmenter.max_vector_distance_subarray(
                S, num_parents, self._dist_func, self.crossover_parent_selection
            )
            parents = [P[i] for i in indices]
        else:
            # Choose the best parents
            parents = P[:num_parents]

        np.random.shuffle(parents)
        parent_prompt = ""
        for parent in parents:
            parent_prompt += f"Parent prompt {len(parents)}: {parent}\n"
        if self.crossover_message is None:
            default_crossover_prompt = SEEPromptAugmenter._load_default_prompt(
                "see_default_crossover.md"
            )
            message = AgentMessage(
                query=default_crossover_prompt,
                context={"parents": parent_prompt},
            )
        else:
            message = self.crossover_message
            if message.context is None:
                message.context = {"parents": parent_prompt}
            else:
                message.context["parents"] = parent_prompt
        message = self.crossover_chain(message, **kwargs)
        return message.responses[-1][1]

    def feedback(
        self, candidate: str, performance: SEEPerformanceTuple, **kwargs
    ) -> str:
        """
        Generate an improved version of the prompt based on the feedback from the examiner.

        Args:
            candidate (str): the original prompt
            **kwargs: arguments for Examiner and Improver chains

        Returns:
            str: the generated prompt
        """
        perf_vec, _ = performance
        wrong_cases = []
        for i, d in enumerate(self.dev_set):
            if perf_vec[i] < 0.5:
                wrong_cases.append(d[0])
                if len(wrong_cases) >= self.num_feedback_wrongcases:
                    break
        wrong_cases = "\n".join(wrong_cases)

        if self.examiner_message is None:
            default_examiner_prompt = SEEPromptAugmenter._load_default_prompt(
                "see_default_examiner.md"
            )
            message = AgentMessage(
                query=default_examiner_prompt,
                context={"candidate": candidate, "wrong_cases": wrong_cases},
            )
        else:
            message = self.examiner_message
            if message.context is None:
                message.context = {"candidate": candidate, "wrong_cases": wrong_cases}
            else:
                message.context["candidate"] = candidate
                message.context["wrong_cases"] = wrong_cases
        message = self.examiner_chain(message, **kwargs)

        feedback_msg = message.responses[-1][1]
        if self.improver_message is None:
            default_improver_prompt = SEEPromptAugmenter._load_default_prompt(
                "see_default_improver.md"
            )
            message = AgentMessage(
                query=default_improver_prompt,
                context={"candidate": candidate, "feedback": feedback_msg},
            )
        else:
            message = self.improver_message
            if message.context is None:
                message.context = {"candidate": candidate, "feedback": feedback_msg}
            else:
                message.context["candidate"] = candidate
                message.context["feedback"] = feedback_msg
        message = self.improver_chain(message, **kwargs)
        return message.responses[-1][1]

    def semantic(self, candidate: str, **kwargs) -> str:
        """
        Generate another prompt with the same semantic meaning and intentions.

        This method works with corresponding semantic_message and semantic_chain attributes.
        In the prompt in semantic_message, it must contain  key 'context.candidate' to parse with the candidate input.

        Args:
            candidate (str): the original prompt
            **kwargs: arguments for semantic chains

        Returns:
            str: the generated prompt
        """
        if self.semantic_message is None:
            message = AgentMessage(
                query="""You are a mutator. Given a prompt, your task is to generate another prompt with the
same semantic meaning and intentions.
# Example:
current prompt: Your mission is to ascertain the sentiment of the provided text and
assign a sentiment label from ['negative', 'positive'].
mutated prompt: Determine the sentiment of the given sentence and assign a label
from ['negative', 'positive'].

## Given ##
current prompt: {context.candidate}
mutated prompt:
""",
                context={"candidate": candidate},
            )
        else:
            message = self.semantic_message
            if message.context is None:
                message.context = {"candidate": candidate}
            else:
                message.context["candidate"] = candidate
        message = self.semantic_chain(message, **kwargs)
        return message.responses[-1][1]

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        P = []  # prompt pool
        S = []  # corresponding performance scores on dev set

        # phase 0: global initialization
        # use lamarckian or sematic to diverse initial prompt
        if isinstance(self.init_data, str):
            while len(P) < self.pool_size_0:
                prompt = self.semantic(self.init_data)
                perf = self._scorer(
                    self.base_chain, prompt, P, S, self.dev_set, **self._scorer_args
                )
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)
        else:
            while len(P) < self.pool_size_0:
                prompt = self.lamarckian(self.init_data)
                perf = self._scorer(
                    self.base_chain, prompt, P, S, self.dev_set, **self._scorer_args
                )
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)

        # phase 1: local feedback operation
        # use feedback to generate new prompt
        t = 0
        k = 0
        old_score = max([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_1:
            P_t = []
            S_t = []
            for prompt, s in zip(P, S):
                new_prompt = self.feedback(prompt, s)
                new_perf = self._scorer(
                    self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
                )
                if new_perf is not None:
                    P_t.append(new_prompt)
                    S_t.append(new_perf)

            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            P, S = SEEPromptAugmenter._sort_pool(
                [*P, *P_t], [*S, *S_t], self.pool_size_1
            )
            new_score = max([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        # phase 2: global fusion operation
        # use eda and crossover to fusion existing prompt and generate new prompt
        t = 0
        k = 0
        old_score = np.mean([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_2:
            new_prompt = self.eda(P, S) if k % 2 else self.crossover(P, S)
            new_perf = self._scorer(
                self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
            )
            if new_perf is not None:
                P.append(new_prompt)
                S.append(new_perf)

            # Performance gain is defined as whether the average performance of the pool is improved
            # Note: we calculate the average score using highest score members and limited by the pool size
            P, S = SEEPromptAugmenter._sort_pool(P, S, self.pool_size_2)
            new_score = np.mean([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        # phase 3: local semantic operation
        # use semantic to generate new prompts
        t = 0
        k = 0
        old_score = max([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_3:
            P_t = []
            S_t = []
            for prompt in P:
                new_prompt = self.semantic(prompt)
                new_perf = self._scorer(
                    self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
                )
                if new_perf is not None:
                    P_t.append(new_prompt)
                    S_t.append(new_perf)

            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            P, S = SEEPromptAugmenter._sort_pool(
                [*P, *P_t], [*S, *S_t], self.pool_size_3
            )
            new_score = max([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        message.query = P[np.argmax([s[1] for s in S])]
        return message


class PromptOptimizationEnv(gym.Env):
    """A Gymnasium environment for prompt optimization.

    This environment allows an agent to select prompt modification actions
    and receive rewards based on the quality of the resulting prompts.

    The environment maintains a current prompt and allows the agent to apply
    prompt augmenters (actions) to modify it. The observation is the embedding
    of the current prompt, and the reward is computed by the reward model.
    The current state is printed to the command line after each step.

    Attributes:
        action_space: Discrete space where each value maps to a prompt augmenter
        observation_space: Box space representing prompt embeddings
        reward_model: Callable that takes a prompt and returns a quality score

    Example:
        ```python
        from aap_core.prompt_augmenter import PromptOptimizationEnv
        import numpy as np

        # Define your augmenters
        augmenters = [
            SimplePromptAugmenter(format="{query} {data}", data_key="context.data"),
            MetaPromptAugmenter(chain=llm_chain)
        ]

        # Define embedding model (e.g., using sentence-transformers)
        def embedding_model(prompt: str) -> np.ndarray:
            # Return embedding vector
            # Example: return embedding_model.encode(prompt)
            return np.random.randn(768).astype(np.float32)

        # Define reward model
        def reward_model(prompt: str) -> float:
            # Return quality score based on prompt
            return 0.0

        # Create environment
        env = PromptOptimizationEnv(
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            initial_prompt="Your initial prompt here",
            max_steps=10
        )

        # Use with Gymnasium API
        obs, info = env.reset()
        action = env.action_space.sample()  # Or use RL agent
        obs, reward, terminated, truncated, info = env.step(action)

        # Or use with RL libraries like Stable-Baselines3
        # from stable_baselines3 import PPO
        # model = PPO("MlpPolicy", env, verbose=1)
        # model.learn(total_timesteps=10000)
        # obs, info = env.reset()
        # for _ in range(100):
        #     action, _ = model.predict(obs)
        #     obs, reward, terminated, truncated, info = env.step(action)

        # Or use with TorchRL (https://docs.pytorch.org/rl/stable/index.html) for DQN training
        # from torchrl.envs import GymEnv, TransformedEnv, StepCounter
        # from torchrl.collectors import SyncDataCollector
        # from torchrl.data import LazyTensorStorage, ReplayBuffer
        # from torchrl.modules import MLP, QValueModule, EGreedyModule
        # from torchrl.objectives import DQNLoss
        # import torch
        #
        # # Wrap the Gymnasium environment for TorchRL
        # env = TransformedEnv(GymEnv(env), StepCounter())
        #
        # # Build Q-value network
        # value_net = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
        # q_value_module = QValueModule(
        #     value_net, in_keys=["observation"], out_keys=["action_value"], spec=env.action_spec
        # )
        #
        # # Add exploration with epsilon-greedy
        # policy = q_value_module
        # exploration_module = EGreedyModule(
        #     eps_init=0.5, eps_final=0.05, annealing_num_steps=10000
        # )
        # policy_explore = torch.nn.Sequential(policy, exploration_module)
        #
        # # Data collector and replay buffer
        # collector = SyncDataCollector(
        #     env,
        #     policy_explore,
        #     frames_per_batch=100,
        #     total_frames=10000,
        #     init_random_frames=500,
        # )
        # replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))
        #
        # # Loss and optimizer
        # loss_module = DQNLoss(value_network=q_value_module, action_space=env.action_spec, delay_value=True)
        # optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
        #
        # # Training loop
        # for data in collector:
        #     replay_buffer.extend(data)
        #     if len(replay_buffer) > 500:
        #         for _ in range(10):
        #             sample = replay_buffer.sample(32)
        #             loss_vals = loss_module(sample.to("cpu"))
        #             loss_vals["loss"].backward()
        #             optimizer.step()
        #             optimizer.zero_grad()
        #         exploration_module.step(data.numel())
        ```
    """

    def __init__(
        self,
        initial_prompt: str,
        augmenters: Sequence[BasePromptAugmenter],
        embedding_model: Callable[[str], np.ndarray],
        reward_model: Callable[[str], float],
        max_steps: int = 10,
        min_embedding_threshold: float = 0.8,
        reward_threshold: float = float("inf"),
        penalty_weight: float = 0.2,
        eps: float = 1e-8,
    ):
        """Initialize the prompt optimization environment.

        Args:
            initial_prompt: The starting prompt for the episode.
            augmenters: List of BasePromptAugmenter objects representing actions.
                       Each augmenter can modify the prompt in different ways.
            embedding_model: Callable that takes a prompt string and returns
                           its embedding as a numpy array.
            reward_model: Callable that takes a prompt string and returns
                        a float indicating prompt quality.
            max_steps: Maximum number of steps per episode.
            min_embedding_threshold: episode terminated if the distance between 2
                consecutive prompt embeddings falls below this threshold.
            reward_threshold: episode terminated if the reward exceeds this threshold.
                Defaults to infinity (no reward-based termination).
            penalty_weight: Weight multiplier for the duplication penalty. The final
                reward is calculated as: base_reward - (duplication_penalty * penalty_weight).
                Higher values penalize duplication more heavily. Defaults to 0.5.
        """
        super().__init__()

        # Validate that at least one augmenter is provided
        assert len(augmenters) > 0, (
            "At least one augmenter must be provided. "
            "An empty list of augmenters results in an empty action space."
        )

        self._initial_prompt = initial_prompt
        self._augmenters = augmenters
        self._embedding_model = embedding_model
        self._reward_model = reward_model
        self.max_steps = max_steps
        self._min_embedding_threshold = min_embedding_threshold
        self._reward_threshold = reward_threshold
        self._penalty_weight = penalty_weight
        self.eps = eps

        # Determine embedding dimension by testing with initial prompt
        self._embedding_dim = len(self._embedding_model(initial_prompt))
        self._prev_embedding = None

        # Action space: discrete space where each value represents an augmenter
        # Actions are in range [0, num_augmenters)
        self.action_space = spaces.Discrete(len(self._augmenters))

        # Observation space: Box space for embedding vectors
        # Using unbounded box since embeddings can have any float values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._embedding_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (prompt embedding).

        Returns:
            The embedding of the current prompt as a numpy array.
        """
        return self._current_embedding.copy()

    def _get_info(self) -> Dict:
        """Get auxiliary information about the current state.

        Returns:
            Dictionary containing current prompt and step count.
        """
        return {
            "current_prompt": self._current_prompt,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "num_augmenters": len(self._augmenters),
        }

    def reset(
        self, *, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (not used currently).

        Returns:
            Tuple of (observation, info) for the initial state.
        """
        super().reset(seed=seed)

        self._current_prompt = self._initial_prompt
        self._current_embedding = self._embedding_model(self._current_prompt).astype(
            np.float32
        )
        self._prev_embedding = self._current_embedding.copy()
        self._step_count = 0
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: The action index (0 to num_augmenters-1) selecting which
                   augmenter to apply to the current prompt.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        elif isinstance(self._augmenters[action], IdentityPromptAugmenter):
            self._step_count += 1
            return self._current_embedding.copy(), 0, True, False, self._get_info()

        try:
            augmenter = self._augmenters[action]
            message = AgentMessage(query=self._current_prompt)
            result_message = augmenter(message)
            self._current_prompt = result_message.query
        except Exception as e:
            # If augmentation fails, keep the current prompt
            warnings.warn(f"Augmenter {action} failed: {e}")

        # Calculate reward with duplication penalty (additive model)
        base_reward = self._reward_model(self._current_prompt)
        duplication_penalty = self._calculate_duplication_penalty(self._current_prompt)
        reward = base_reward - (duplication_penalty * self._penalty_weight)
        self._current_embedding = self._embedding_model(self._current_prompt).astype(
            np.float32
        )
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        terminated = False
        terminated_reason = None

        # Check embedding distance between consecutive prompts
        if self._prev_embedding is not None:
            cosine_sim = np.dot(self._current_embedding, self._prev_embedding) / (
                np.linalg.norm(self._current_embedding)
                * np.linalg.norm(self._prev_embedding)
                + self.eps
            )
            if cosine_sim < self._min_embedding_threshold:
                terminated = True
                terminated_reason = f"Cosine similarity {cosine_sim:.6f} < threshold {self._min_embedding_threshold}"

        # Check if reward exceeds threshold
        if not terminated and self._reward_threshold < float("inf"):
            if reward > self._reward_threshold:
                terminated = True
                terminated_reason = (
                    f"Reward {reward:.4f} > threshold {self._reward_threshold}"
                )

        # Update previous embedding for next step
        self._prev_embedding = self._current_embedding.copy()
        observation = self._get_observation()
        info = self._get_info()
        info["terminated_reason"] = terminated_reason
        info["duplication_penalty"] = duplication_penalty
        # self._print_state()

        return observation, reward, terminated, truncated, info

    def _calculate_duplication_penalty(self, text: str) -> float:
        """Calculate a penalty factor proportional to the square root of the total
        length of duplicate text using a Bloom filter for efficient duplicate detection.

        This method tokenizes the text into sentences and uses a Bloom filter to
        probabilistically identify duplicate sentences. The penalty is calculated
        as the square root of the total character length of all duplicate sentences,
        making the penalty grow with the severity of duplication.

        Args:
            text: The text to analyze for duplicates.

        Returns:
            A float representing the duplication penalty, proportional to
            sqrt(total_duplicate_chars).
        """
        if not text or not text.strip():
            return 0.0

        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text.strip())
        if len(sentences) <= 1:
            return 0.0

        # Create a Bloom filter for efficient duplicate detection
        # Expected items = number of sentences, FPR = 0.01 for low false positive rate
        bloom = Bloom(
            expected_items=len(sentences),
            false_positive_rate=0.01,
        )

        # Count duplicates and accumulate their total character length
        total_dup_chars = 0
        for sent in sentences:
            normalized = sent.strip().lower()
            if normalized in bloom:
                # Sentence already in Bloom filter - it's a duplicate
                total_dup_chars += len(sent)
            else:
                # Not seen before - add to Bloom filter
                bloom.add(normalized)

        # Penalty is proportional to the square root of total duplicate text length
        penalty = np.sqrt(total_dup_chars)
        return penalty

    def _print_state(self) -> None:
        """Print the current environment state to the command line."""
        print("=" * 60)
        print(f"Step: {self._step_count} / {self.max_steps}")
        print(f"Current Prompt: {self._current_prompt}")
        print(f"Quality: {self._reward_model(self._current_prompt):.4f}")
        print("=" * 60)


class RLPromptAugmenter(BasePromptAugmenter):
    """RL-based prompt augmenter using policy gradient methods.

    **RL problem fomulation**
    - State/observation: the current prompt (represented by its embedding vector)
    - Action: selecting one of the available prompt augmenters to apply
    - Reward: there are 2 types of reward signals in this task:
        1. The quality score of the resulting prompt after applying the selected augmenter, which is computed by the reward model in the environment.
        2. The accuracy of the resulting prompt on the dev set.
    - Transition: applying an augmenter modifies the current prompt and leads to a new state (new prompt embedding).
    - Environment: the PromptOptimizationEnv class

    Translate to gymnasium:
    - State: gymnasium.spaces.Box (n dimensional vector)
    - Action: gymnasium.spaces.Discrete (number of augmenters)
    """

    model_config = {"arbitrary_types_allowed": True}
    env: PromptOptimizationEnv = Field(
        ..., description="The prompt optimization environment"
    )
    policy_model: BasePolicy = Field(..., description="The trained policy model")

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        """Use the trained policy to iteratively apply augmenters and update the prompt.

        The policy model takes the current prompt embedding as input and outputs
        action logits. We sample actions from the policy and apply the corresponding
        augmenters until max_steps is reached.

        The initial prompt is taken from the 'message' object's query field.

        Args:
            message: The input AgentMessage containing the initial prompt.
            **kwargs: Additional arguments passed to the environment step.

        Returns:
            The updated AgentMessage with the optimized prompt.
        """
        # Set the initial prompt from the message object
        self.env._initial_prompt = message.query
        self.env._current_prompt = message.query

        # Reset environment with the current prompt
        obs, info = self.env.reset()
        step_count = 0

        while step_count < self.env.max_steps:
            # Convert observation to tensor with proper shape for policy model
            # obs shape: (embedding_dim,) -> (1, embedding_dim) for batch=1
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Get action from policy (no gradient needed for inference)
            with torch.no_grad():
                logits = self.policy_model(obs_tensor)
                # Extract action from policy output using helper method
                action, action_log_prob = self.policy_model.get_action(logits)

            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            step_count += 1

            # Stop if episode terminated early
            if terminated or truncated:
                break

        message.query = info["current_prompt"]
        return message
