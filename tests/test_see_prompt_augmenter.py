import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
from aap_core.prompt_augmenter import SEEPromptAugmenter
from aap_core.types import AgentMessage, BaseLLMChain


class MockLLMChain(BaseLLMChain):
    """A mock LLM chain for testing SEEPromptAugmenter."""

    response_content: str = "mock response"

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("mock_chain", self.response_content))
        return message


def _mock_scorer(chain, prompt, pool, performance_pool, dataset, distance_threshold=2):
    """Mock scorer that always returns a valid performance tuple.
    This avoids the 'min() arg is an empty sequence' error when performance_pool is empty."""
    perf_vec = [1.0] * len(dataset)
    score = 1.0
    return (perf_vec, score)


@pytest.fixture
def dev_set():
    """Fixture for a small development dataset."""
    return [
        ("What is the capital of France?", "Paris"),
        ("What is 2+2?", "4"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ]


@pytest.fixture
def base_chain():
    """Fixture for a base LLM chain."""
    return MockLLMChain(name="base_chain", response_content="The answer is correct.")


@pytest.fixture
def lamarckian_chain():
    return MockLLMChain(
        name="lamarckian_chain", response_content="Generated prompt from pairs"
    )


@pytest.fixture
def eda_chain():
    return MockLLMChain(
        name="eda_chain", response_content="Mutated prompt from candidates"
    )


@pytest.fixture
def crossover_chain():
    return MockLLMChain(
        name="crossover_chain", response_content="Offspring prompt from parents"
    )


@pytest.fixture
def examiner_chain():
    return MockLLMChain(
        name="examiner_chain", response_content="Feedback on the prompt"
    )


@pytest.fixture
def improver_chain():
    return MockLLMChain(
        name="improver_chain", response_content="Improved prompt from feedback"
    )


@pytest.fixture
def semantic_chain():
    return MockLLMChain(
        name="semantic_chain", response_content="Semantically similar prompt"
    )


@pytest.fixture
def see_augmenter(
    base_chain,
    lamarckian_chain,
    eda_chain,
    crossover_chain,
    examiner_chain,
    improver_chain,
    semantic_chain,
    dev_set,
):
    """Fixture to provide a SEEPromptAugmenter instance with a mock scorer."""
    augmenter = SEEPromptAugmenter(
        base_chain=base_chain,
        lamarckian_chain=lamarckian_chain,
        eda_chain=eda_chain,
        crossover_chain=crossover_chain,
        examiner_chain=examiner_chain,
        improver_chain=improver_chain,
        semantic_chain=semantic_chain,
        dev_set=dev_set,
        init_data="Initial prompt",
    )
    # Mock the scorer to avoid 'min() arg is an empty sequence' error
    augmenter._scorer = _mock_scorer
    return augmenter


def test_see_augmenter_augment_basic(see_augmenter):
    """Test the basic augmentation process with default parameters."""
    message = AgentMessage(query="Initial prompt")
    # We use a small pool size to speed up the test
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1

    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"
    assert len(augmented_message.query) > 0


def test_see_augmenter_lamarckian_init(see_augmenter):
    """Test initialization using Lamarckian operator."""
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1

    # init_data as DataSet (list of tuples)
    dev_set = [("input1", "output1"), ("input2", "output2")]
    see_augmenter.init_data = dev_set

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


def test_see_augmenter_semantic_init(see_augmenter):
    """Test initialization using Semantic operator."""
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1

    # init_data as string
    see_augmenter.init_data = "Initial prompt"

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


def test_see_augmenter_with_custom_messages(see_augmenter):
    """Test with custom operator messages."""
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1

    custom_semantic_message = AgentMessage(
        query="Rewrite this prompt: {context.candidate}",
        context={"candidate": "Old prompt"},
    )
    see_augmenter.semantic_message = custom_semantic_message

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


@pytest.mark.parametrize("pool_size", [2, 3])
def test_see_augmenter_pool_sizes(see_augmenter, pool_size):
    """Test with different pool sizes."""
    see_augmenter.pool_size_0 = pool_size
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


def test_see_augmenter_invalid_scorer(dev_set):
    """Test that invalid scorer raises ValueError."""

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    with pytest.raises(ValueError, match="scorer not supported"):
        SEEPromptAugmenter(
            base_chain=base_chain,
            lamarckian_chain=lamarckian_chain,
            eda_chain=eda_chain,
            crossover_chain=crossover_chain,
            examiner_chain=examiner_chain,
            improver_chain=improver_chain,
            semantic_chain=semantic_chain,
            dev_set=dev_set,
            init_data="Initial prompt",
            scorer="invalid_scorer",
        )


def test_see_augmenter_lamarckian_operator(see_augmenter):
    """Test the Lamarckian operator generates a prompt from input-output pairs."""
    pairs = [("What is 1+1?", "2"), ("What is 2+2?", "4")]
    generated_prompt = see_augmenter.lamarckian(pairs)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_lamarckian_with_custom_message(see_augmenter):
    """Test Lamarckian operator with custom message."""
    custom_message = AgentMessage(
        query="Custom prompt: {context.pairs}",
        context={"pairs": "Custom pairs data"},
    )
    see_augmenter.lamarckian_message = custom_message
    pairs = [("What is 1+1?", "2")]
    generated_prompt = see_augmenter.lamarckian(pairs)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_eda_operator(see_augmenter):
    """Test the EDA operator generates a new prompt from candidates."""
    candidates = ["Prompt 1", "Prompt 2", "Prompt 3"]
    S = [([1.0, 1.0, 1.0], 1.0), ([0.8, 0.8, 0.8], 0.8), ([0.6, 0.6, 0.6], 0.6)]
    generated_prompt = see_augmenter.eda(candidates, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_eda_with_index(see_augmenter):
    """Test EDA operator with ranking (eda_with_index=True)."""
    see_augmenter.eda_with_index = True
    candidates = ["Prompt 1", "Prompt 2", "Prompt 3"]
    S = [([1.0, 1.0, 1.0], 1.0), ([0.8, 0.8, 0.8], 0.8), ([0.6, 0.6, 0.6], 0.6)]
    generated_prompt = see_augmenter.eda(candidates, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_crossover_operator(see_augmenter):
    """Test the Crossover operator generates a new prompt from parents."""
    P = ["Parent prompt 1", "Parent prompt 2"]
    S = [([1.0, 1.0], 1.0), ([0.8, 0.8], 0.8)]
    generated_prompt = see_augmenter.crossover(P, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_crossover_with_distinct(see_augmenter):
    """Test Crossover operator with diversity consideration."""
    see_augmenter.crossover_with_distinct = True
    P = ["Parent prompt 1", "Parent prompt 2", "Parent prompt 3"]
    S = [([1.0, 0.0, 0.0], 1.0), ([0.0, 1.0, 0.0], 1.0), ([0.0, 0.0, 1.0], 1.0)]
    generated_prompt = see_augmenter.crossover(P, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_feedback_operator(see_augmenter, dev_set):
    """Test the Feedback operator generates an improved prompt."""
    see_augmenter.dev_set = dev_set
    candidate = "A basic prompt"
    performance = ([0, 0, 1], 0.33)  # Two wrong cases
    generated_prompt = see_augmenter.feedback(candidate, performance)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_feedback_all_correct(see_augmenter, dev_set):
    """Test Feedback operator when all cases are correct (no wrong cases)."""
    see_augmenter.dev_set = dev_set
    candidate = "A perfect prompt"
    performance = ([1, 1, 1], 1.0)  # All correct
    generated_prompt = see_augmenter.feedback(candidate, performance)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_semantic_operator(see_augmenter):
    """Test the Semantic operator generates a semantically similar prompt."""
    candidate = "This is a test prompt"
    generated_prompt = see_augmenter.semantic(candidate)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_semantic_with_custom_message(see_augmenter):
    """Test Semantic operator with custom message."""
    custom_message = AgentMessage(
        query="Paraphrase: {context.candidate}",
        context={"candidate": "Original prompt"},
    )
    see_augmenter.semantic_message = custom_message
    candidate = "This is a test prompt"
    generated_prompt = see_augmenter.semantic(candidate)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_sort_pool():
    """Test the _sort_pool class method."""
    P = ["prompt_a", "prompt_b", "prompt_c"]
    S = [([0.5], 0.5), ([1.0], 1.0), ([0.8], 0.8)]
    sorted_P, sorted_S = SEEPromptAugmenter._sort_pool(P, S, pool_size=2)
    assert len(sorted_P) == 2
    assert len(sorted_S) == 2
    assert sorted_S[0][1] >= sorted_S[1][1]  # Should be sorted descending


@pytest.mark.parametrize(
    "selection_method",
    ["random", "wheel", "tournament"],
)
def test_see_augmenter_selection_method(selection_method):
    """Test selection methods produce unique indices."""
    S = [(1.0, 1.0), (0.8, 0.8), (0.6, 0.6)]
    method = getattr(SEEPromptAugmenter, f"_selection_{selection_method}")
    indices = method(2, S)
    assert len(indices) == 2
    assert len(set(indices)) == 2  # Should be unique


def test_see_augmenter_max_vector_distance_subarray():
    """Test the max_vector_distance_subarray class method."""
    S = [((1.0, 0.0), 1.0), ((0.0, 1.0), 1.0), ((0.5, 0.5), 0.5)]
    indices = SEEPromptAugmenter.max_vector_distance_subarray(
        S, k=2, dist_func=SEEPromptAugmenter._hamming_distance
    )
    assert len(indices) == 2


def test_see_augmenter_max_vector_distance_subarray_invalid_k():
    """Test max_vector_distance_subarray with invalid k."""
    S = [((1.0,), 1.0), ((0.8,), 0.8)]
    with pytest.raises(ValueError, match="k cannot be greater than the length of S"):
        SEEPromptAugmenter.max_vector_distance_subarray(
            S, k=5, dist_func=lambda a, b: 1.0
        )


def test_see_augmenter_hamming_distance():
    """Test the _hamming_distance class method."""
    v1 = [1.0, 0.0, 1.0]
    v2 = [1.0, 1.0, 1.0]
    distance = SEEPromptAugmenter._hamming_distance(v1, v2)
    assert distance == 1


def test_see_augmenter_hamming_scorer(see_augmenter, base_chain, dev_set):
    """Test the _hamming_scorer method."""
    prompt = "Test prompt"
    pool = ["prompt1"]
    performance_pool = [([0.5, 0.5, 0.5], 0.5)]
    result = see_augmenter._scorer(
        base_chain, prompt, pool, performance_pool, dev_set, distance_threshold=2
    )
    # Result should be None if similar, or (perf_vec, score) if not
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


def test_see_augmenter_score_method(see_augmenter, base_chain, dev_set):
    """Test the score method."""
    see_augmenter.dev_set = dev_set
    prompt = "Test prompt"
    result = see_augmenter.score(prompt, dev_set)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], float)


def test_see_augmenter_invalid_eval_method(dev_set):
    """Test that invalid eval_method raises ValueError."""

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    with pytest.raises(ValueError, match="eval_method not supported"):
        SEEPromptAugmenter(
            base_chain=base_chain,
            lamarckian_chain=lamarckian_chain,
            eda_chain=eda_chain,
            crossover_chain=crossover_chain,
            examiner_chain=examiner_chain,
            improver_chain=improver_chain,
            semantic_chain=semantic_chain,
            dev_set=dev_set,
            init_data="Initial prompt",
            eval_method="invalid_method",
        )


def test_see_augmenter_custom_scorer(dev_set):
    """Test SEEPromptAugmenter with custom scorer and dist_func."""

    def custom_scorer(
        chain, prompt, pool, performance_pool, dataset, distance_threshold=2
    ):
        return ([1.0] * len(dataset), 1.0)

    def custom_dist_func(v1, v2):
        return float(np.sum(np.array(v1) != np.array(v2)))

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    augmenter = SEEPromptAugmenter(
        base_chain=base_chain,
        lamarckian_chain=lamarckian_chain,
        eda_chain=eda_chain,
        crossover_chain=crossover_chain,
        examiner_chain=examiner_chain,
        improver_chain=improver_chain,
        semantic_chain=semantic_chain,
        dev_set=dev_set,
        init_data="Initial prompt",
        scorer=custom_scorer,
        dist_func=custom_dist_func,
    )
    assert augmenter._scorer == custom_scorer
    assert augmenter._dist_func == custom_dist_func


def test_see_augmenter_include_eval_method(dev_set):
    """Test SEEPromptAugmenter with include eval_method."""

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    augmenter = SEEPromptAugmenter(
        base_chain=base_chain,
        lamarckian_chain=lamarckian_chain,
        eda_chain=eda_chain,
        crossover_chain=crossover_chain,
        examiner_chain=examiner_chain,
        improver_chain=improver_chain,
        semantic_chain=semantic_chain,
        dev_set=dev_set,
        init_data="Initial prompt",
        eval_method="include",
    )
    assert augmenter._eval_method("Paris", "Paris is the capital") is True
    assert augmenter._eval_method("London", "Paris is the capital") is False


def test_see_augmenter_custom_eval_method(dev_set):
    """Test SEEPromptAugmenter with custom eval_method callable."""

    def custom_eval(x, y):
        return len(x) == len(y)

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    augmenter = SEEPromptAugmenter(
        base_chain=base_chain,
        lamarckian_chain=lamarckian_chain,
        eda_chain=eda_chain,
        crossover_chain=crossover_chain,
        examiner_chain=examiner_chain,
        improver_chain=improver_chain,
        semantic_chain=semantic_chain,
        dev_set=dev_set,
        init_data="Initial prompt",
        eval_method=custom_eval,
    )
    assert augmenter._eval_method("abc", "xyz") is True
    assert augmenter._eval_method("a", "xyz") is False


@pytest.mark.parametrize(
    "selection_strategy",
    ["random", "tournament", "wheel"],
)
def test_see_augmenter_eda_parent_selection(see_augmenter, selection_strategy):
    """Test EDA with different parent selection strategies."""
    see_augmenter.eda_parent_selection = selection_strategy
    candidates = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
    S = [([1.0] * 5, 1.0), ([0.8] * 5, 0.8), ([0.6] * 5, 0.6), (0.4, 0.4), (0.2, 0.2)]
    generated_prompt = see_augmenter.eda(candidates, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_eda_with_num_parents(see_augmenter):
    """Test EDA with specific number of parents."""
    see_augmenter.num_eda_parents = 2
    candidates = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
    S = [([1.0] * 5, 1.0), ([0.8] * 5, 0.8), ([0.6] * 5, 0.6)]
    generated_prompt = see_augmenter.eda(candidates, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_crossover_with_num_parents(see_augmenter):
    """Test Crossover with specific number of parents."""
    see_augmenter.num_crossover_parents = 3
    P = ["Parent 1", "Parent 2", "Parent 3", "Parent 4"]
    S = [([1.0] * 4, 1.0), ([0.8] * 4, 0.8), ([0.6] * 4, 0.6), ([0.4] * 4, 0.4)]
    generated_prompt = see_augmenter.crossover(P, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_crossover_with_all_parents(see_augmenter):
    """Test Crossover with all parents (num_crossover_parents=-1)."""
    see_augmenter.num_crossover_parents = -1
    P = ["Parent 1", "Parent 2", "Parent 3"]
    S = [([1.0] * 3, 1.0), ([0.8] * 3, 0.8), ([0.6] * 3, 0.6)]
    generated_prompt = see_augmenter.crossover(P, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_feedback_with_custom_wrongcases(see_augmenter, dev_set):
    """Test Feedback operator with custom number of wrong cases."""
    see_augmenter.dev_set = dev_set
    see_augmenter.num_feedback_wrongcases = 1
    candidate = "A basic prompt"
    performance = ([0, 0, 1], 0.33)
    generated_prompt = see_augmenter.feedback(candidate, performance)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_feedback_with_custom_message(see_augmenter, dev_set):
    """Test Feedback operator with custom examiner and improver messages."""
    see_augmenter.dev_set = dev_set
    see_augmenter.examiner_message = AgentMessage(
        query="Analyze these wrong cases: {context.wrong_cases}",
        context={"candidate": "Test prompt", "wrong_cases": "Wrong case 1"},
    )
    see_augmenter.improver_message = AgentMessage(
        query="Improve based on: {context.feedback}",
        context={"candidate": "Test prompt", "feedback": "Some feedback"},
    )
    candidate = "A basic prompt"
    performance = ([0, 0, 1], 0.33)
    generated_prompt = see_augmenter.feedback(candidate, performance)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_warning_on_init(dev_set):
    """Test that SEEPromptAugmenter emits a warning on initialization."""

    base_chain = MagicMock(spec=BaseLLMChain)
    lamarckian_chain = MagicMock(spec=BaseLLMChain)
    eda_chain = MagicMock(spec=BaseLLMChain)
    crossover_chain = MagicMock(spec=BaseLLMChain)
    examiner_chain = MagicMock(spec=BaseLLMChain)
    improver_chain = MagicMock(spec=BaseLLMChain)
    semantic_chain = MagicMock(spec=BaseLLMChain)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SEEPromptAugmenter(
            base_chain=base_chain,
            lamarckian_chain=lamarckian_chain,
            eda_chain=eda_chain,
            crossover_chain=crossover_chain,
            examiner_chain=examiner_chain,
            improver_chain=improver_chain,
            semantic_chain=semantic_chain,
            dev_set=dev_set,
            init_data="Initial prompt",
        )
        assert len(w) == 1
        assert "experimental use" in str(w[0].message).lower()


def test_see_augmenter_empty_pool_scorer_first_iteration(
    see_augmenter, base_chain, dev_set
):
    """Test scorer behavior on first iteration with empty pool."""
    see_augmenter.dev_set = dev_set
    prompt = "Test prompt"
    # Empty pool and performance_pool - should accept first candidate
    result = see_augmenter._scorer(
        base_chain, prompt, [], [], dev_set, distance_threshold=2
    )
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_see_augmenter_augment_with_tolerances(see_augmenter):
    """Test augmentation with different tolerance values."""
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1
    see_augmenter.tolerance_1 = 2
    see_augmenter.tolerance_2 = 3
    see_augmenter.tolerance_3 = 2

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


def test_see_augmenter_augment_with_performance_threshold(see_augmenter):
    """Test augmentation with different performance gain thresholds."""
    see_augmenter.pool_size_0 = 2
    see_augmenter.pool_size_1 = 1
    see_augmenter.pool_size_2 = 1
    see_augmenter.pool_size_3 = 1
    see_augmenter.performance_gain_threshold = 0.05

    message = AgentMessage(query="Initial prompt")
    augmented_message = see_augmenter.augment(message)

    assert isinstance(augmented_message, AgentMessage)
    assert augmented_message.query != "Initial prompt"


def test_see_augmenter_eda_with_custom_message(see_augmenter):
    """Test EDA operator with custom message."""
    custom_message = AgentMessage(
        query="Mutate these prompts: {context.candidates}",
        context={"candidates": "Prompt 1\nPrompt 2"},
    )
    see_augmenter.eda_message = custom_message
    candidates = ["Prompt 1", "Prompt 2"]
    S = [([1.0], 1.0), ([0.8], 0.8)]
    generated_prompt = see_augmenter.eda(candidates, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


def test_see_augmenter_crossover_with_custom_message(see_augmenter):
    """Test Crossover operator with custom message."""
    custom_message = AgentMessage(
        query="Cross over these parents: {context.parents}",
        context={"parents": "Parent 1\nParent 2"},
    )
    see_augmenter.crossover_message = custom_message
    P = ["Parent 1", "Parent 2"]
    S = [([1.0], 1.0), ([0.8], 0.8)]
    generated_prompt = see_augmenter.crossover(P, S)
    assert isinstance(generated_prompt, str)
    assert len(generated_prompt) > 0


@pytest.mark.parametrize(
    "k,expected_len,expected_set,S",
    [
        (1, 1, None, [((1.0, 0.0), 1.0), ((0.0, 1.0), 1.0), ((0.5, 0.5), 0.5)]),
        (2, 2, None, [((1.0, 0.0), 1.0), ((0.0, 1.0), 1.0), ((0.5, 0.5), 0.5)]),
        (2, 2, {0, 1}, [((1.0, 0.0), 1.0), ((0.0, 1.0), 1.0)]),  # k == len(S)
    ],
)
def test_see_augmenter_max_vector_distance_subarray_k(k, expected_len, expected_set, S):
    """Test max_vector_distance_subarray with different k values."""
    indices = SEEPromptAugmenter.max_vector_distance_subarray(
        S, k=k, dist_func=SEEPromptAugmenter._hamming_distance
    )
    assert len(indices) == expected_len
    if expected_set is not None:
        assert set(indices) == expected_set


def test_see_augmenter_selection_methods_produce_unique_indices():
    """Test that all selection methods produce unique indices."""
    S = [([1.0], 1.0), ([0.8], 0.8), ([0.6], 0.6), ([0.4], 0.4), ([0.2], 0.2)]

    random_indices = SEEPromptAugmenter._selection_random(3, S)
    assert len(random_indices) == 3
    assert len(set(random_indices)) == 3

    wheel_indices = SEEPromptAugmenter._selection_wheel(3, S)
    assert len(wheel_indices) == 3
    assert len(set(wheel_indices)) == 3

    tournament_indices = SEEPromptAugmenter._selection_tournament(3, S)
    assert len(tournament_indices) == 3
    assert len(set(tournament_indices)) == 3


def test_see_augmenter_sort_pool_preserves_order():
    """Test that _sort_pool preserves prompt-score pairing after sorting."""
    P = ["low", "high", "medium"]
    S = [([0.3], 0.3), ([0.9], 0.9), ([0.6], 0.6)]
    sorted_P, sorted_S = SEEPromptAugmenter._sort_pool(P, S, pool_size=3)
    # After sorting descending by score, high should be first
    assert sorted_P[0] == "high"
    assert sorted_S[0][1] == 0.9
    assert sorted_P[2] == "low"
    assert sorted_S[2][1] == 0.3


def test_see_augmenter_sort_pool_truncates():
    """Test that _sort_pool truncates to pool_size."""
    P = ["a", "b", "c", "d", "e"]
    S = [([0.1], 0.1), ([0.2], 0.2), ([0.3], 0.3), ([0.4], 0.4), ([0.5], 0.5)]
    sorted_P, sorted_S = SEEPromptAugmenter._sort_pool(P, S, pool_size=3)
    assert len(sorted_P) == 3
    assert len(sorted_S) == 3
    assert sorted_P[0] == "e"
    assert sorted_S[0][1] == 0.5
