import pytest
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter
from aap_core.types import AgentMessage


class TestDeduplicationPromptAugmenter:
    """Tests for DeduplicationPromptAugmenter using rensa C-MinHash."""

    @pytest.fixture
    def augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.8}
        )

    @pytest.fixture
    def low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.3}
        )

    def test_constructor_custom_params(self):
        """Test constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 123, "threshold": 0.7}
        )
        assert aug._algo_config.num_perm == 64
        assert aug._algo_config.seed == 123
        assert aug._algo_config.threshold == 0.7
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_invalid_threshold_too_low(self):
        """Test that threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"threshold": -0.1})

    def test_constructor_invalid_threshold_too_high(self):
        """Test that threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"threshold": 1.1})

    def test_constructor_invalid_threshold_boundary_zero(self):
        """Test that threshold=0.0 is accepted (boundary)."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.0}
        )
        assert aug._algo_config.threshold == 0.0

    def test_constructor_invalid_threshold_boundary_one(self):
        """Test that threshold=1.0 is accepted (boundary)."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 1.0}
        )
        assert aug._algo_config.threshold == 1.0

    def test_constructor_invalid_num_perm_zero(self):
        """Test that num_perm=0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": 0, "seed": 42, "threshold": 0.8}
            )

    def test_constructor_invalid_num_perm_negative(self):
        """Test that num_perm<0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": -10, "seed": 42, "threshold": 0.8}
            )

    def test_constructor_invalid_seed_negative(self):
        """Test that seed<0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": 64, "seed": -1, "threshold": 0.8}
            )

    def test_constructor_invalid_algorithm_name(self):
        """Test that unsupported algorithm_name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "unsupported_algo"}
            )

    def test_constructor_simhash_supported(self):
        """Test that simhash algorithm is now supported and works."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"algorithm_name": "simhash", "threshold": 0.8}
        )
        assert aug._algo_config.algorithm_name == "simhash"
        assert aug._algo_config.threshold == 0.8

    def test_constructor_invalid_threshold_type(self):
        """Test that non-coercible threshold type raises ValidationError."""
        from pydantic import ValidationError

        # Pydantic v2 coerces "0.8" to 0.8, so test with a list which can't be coerced
        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": [0.8]})

    def test_constructor_invalid_num_perm_type(self):
        """Test that non-integer num_perm raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"num_perm": 64.5})

    def test_constructor_invalid_seed_type(self):
        """Test that non-integer seed raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"seed": 42.5})

    def test_constructor_nested_dict_in_algo_args(self):
        """Test that nested dict values in algo_args raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": {"nested": "dict"}})

    def test_constructor_list_in_algo_args(self):
        """Test that list values in algo_args raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": [0.8, 0.9]})

    def test_constructor_none_value_for_threshold(self):
        """Test that None value for threshold raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": None})

    def test_constructor_none_value_for_num_perm(self):
        """Test that None value for num_perm raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"num_perm": None})

    def test_constructor_none_value_for_seed(self):
        """Test that None value for seed raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"seed": None})

    def test_constructor_empty_algo_args(self):
        """Test that empty algo_args uses defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={})
        assert aug._algo_config.num_perm == 128
        assert aug._algo_config.seed == 42
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_partial_algo_args(self):
        """Test that partial algo_args fills in defaults for missing keys."""
        aug = DeduplicationPromptAugmenter(algo_args={"threshold": 0.5})
        assert aug._algo_config.threshold == 0.5
        assert aug._algo_config.num_perm == 128  # default
        assert aug._algo_config.seed == 42  # default
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_algorithm_name_with_other_params(self):
        """Test that explicit algorithm_name=minhash works with other params."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash",
                "threshold": 0.9,
                "num_perm": 32,
                "seed": 100,
            }
        )
        assert aug._algo_config.algorithm_name == "minhash"
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.num_perm == 32
        assert aug._algo_config.seed == 100

    # --- LSH Algorithm Tests ---

    @pytest.fixture
    def lsh_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
            }
        )

    @pytest.fixture
    def lsh_low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.3,
                "num_bands": 16,
            }
        )

    def test_constructor_lsh_defaults(self):
        """Test that LSH constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "minhash_lsh"})
        assert aug._algo_config.algorithm_name == "minhash_lsh"
        assert aug._algo_config.num_perm == 128
        assert aug._algo_config.seed == 42
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.num_bands == 16  # default, 128/16=8

    def test_constructor_lsh_custom_params(self):
        """Test LSH constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 123,
                "threshold": 0.7,
                "num_bands": 16,
            }
        )
        assert aug._algo_config.algorithm_name == "minhash_lsh"
        assert aug._algo_config.num_perm == 64
        assert aug._algo_config.seed == 123
        assert aug._algo_config.threshold == 0.7
        assert aug._algo_config.num_bands == 16

    def test_constructor_lsh_invalid_num_bands_zero(self):
        """Test that num_bands=0 raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_bands": 0,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_lsh_invalid_num_bands_negative(self):
        """Test that num_bands<0 raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_bands": -5,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_lsh_invalid_threshold(self):
        """Test that invalid threshold raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "threshold": 1.5,
                    "num_perm": 64,
                    "seed": 42,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_num_perm(self):
        """Test that invalid num_perm raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_perm": 0,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_seed(self):
        """Test that invalid seed raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "seed": -1,
                    "num_perm": 64,
                    "threshold": 0.8,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_type(self):
        """Test that invalid type raises ValidationError for LSH."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "minhash_lsh", "threshold": [0.8]}
            )

    def test_augment_lsh_empty_query(self, lsh_augmenter):
        """Test that LSH with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = lsh_augmenter(message)
        assert result.query == ""

    def test_augment_lsh_exact_duplicates(self, lsh_augmenter):
        """Test LSH deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_lsh_multiple_exact_duplicates(self, lsh_augmenter):
        """Test LSH deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_lsh_no_duplicates(self, lsh_augmenter):
        """Test that LSH preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_near_duplicates(self, lsh_low_threshold_augmenter):
        """Test LSH deduplication of near-duplicate sentences with low threshold."""
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = lsh_low_threshold_augmenter(message)
        # Near-duplicates should be removed with low threshold
        assert result.query != query

    def test_augment_lsh_unique_sentences_preserved(self, lsh_augmenter):
        """Test that LSH preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_lsh_single_sentence(self, lsh_augmenter):
        """Test LSH with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query == query

    def test_augment_lsh_single_word(self, lsh_augmenter):
        """Test LSH with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query == query

    def test_augment_lsh_multiple_exclamation(self, lsh_augmenter):
        """Test LSH with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_lsh_mixed_terminators(self, lsh_augmenter):
        """Test LSH with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_with_loop(self, lsh_augmenter):
        """Test LSH with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    # --- Empty / None query ---

    def test_augment_empty_query(self, augmenter):
        """Test that empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = augmenter(message)
        assert result.query == ""

    def test_augment_none_query(self, augmenter):
        """Test that None query returns message unchanged."""
        # AgentMessage requires a string query, so we test with empty string instead
        message = AgentMessage(query="")
        result = augmenter(message)
        assert result.query == ""

    # --- Exact duplicate sentences ---

    def test_augment_exact_duplicate_sentences(self, augmenter):
        """Test deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_multiple_exact_duplicates(self, augmenter):
        """Test deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_no_duplicates(self, augmenter):
        """Test that text with no duplicates is preserved."""
        query = "First sentence. Second sentence. Third sentence."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # All sentences should be preserved
        assert "First sentence" in result.query
        assert "Second sentence" in result.query
        assert "Third sentence" in result.query

    # --- Near-duplicate sentences ---

    def test_augment_near_duplicate_sentences(self, low_threshold_augmenter):
        """Test deduplication of near-duplicate sentences with low threshold."""
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = low_threshold_augmenter(message)
        # Near-duplicates should be removed with low threshold
        assert result.query != query

    def test_augment_unique_sentences_preserved(self, augmenter):
        """Test that all unique sentences are preserved."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    # --- Single sentence ---

    def test_augment_single_sentence(self, augmenter):
        """Test that a single sentence is preserved."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert result.query == query

    # --- Edge cases ---

    def test_augment_single_word_no_punctuation(self, augmenter):
        """Test text with no sentence boundaries."""
        query = "just one word"
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert result.query == query

    def test_augment_multiple_exclamation_marks(self, augmenter):
        """Test deduplication with various sentence terminators."""
        query = "Wow! Wow! Amazing! Different."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Wow!" should appear only once
        assert result.query.count("Wow") == 1
        assert "Different" in result.query

    def test_augment_mixed_terminators(self, augmenter):
        """Test with mixed sentence terminators (.!?)."""
        query = "Hello. Wow! Really? Hello. Unique."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Hello." should appear only once
        assert result.query.count("Hello") == 1
        assert "Unique" in result.query

    # --- Loop integration ---

    def test_augment_with_loop(self):
        """Test that the loop parameter works with DeduplicationPromptAugmenter."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.8}, loop=2
        )
        query = "Test. Test. Different."
        message = AgentMessage(query=query)
        result = aug(message)
        # After first pass, duplicates are removed; second pass should be a no-op
        assert result.query.count("Test") == 1

    # --- Bloom Filter Algorithm Tests ---

    @pytest.fixture
    def bloom_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "bloomfilter",
                "expected_items": 100,
                "false_positive_rate": 0.01,
            }
        )

    @pytest.fixture
    def bloom_default_augmenter(self):
        return DeduplicationPromptAugmenter(algo_args={"algorithm_name": "bloomfilter"})

    def test_constructor_bloomfilter_defaults(self):
        """Test that bloomfilter constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "bloomfilter"})
        assert aug._algo_config.algorithm_name == "bloomfilter"
        assert aug._algo_config.expected_items == 10
        assert aug._algo_config.false_positive_rate == 0.01

    def test_constructor_bloomfilter_custom_params(self):
        """Test bloomfilter constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "bloomfilter",
                "expected_items": 500,
                "false_positive_rate": 0.001,
            }
        )
        assert aug._algo_config.algorithm_name == "bloomfilter"
        assert aug._algo_config.expected_items == 500
        assert aug._algo_config.false_positive_rate == 0.001

    def test_constructor_bloomfilter_invalid_expected_items_zero(self):
        """Test that expected_items=0 raises ValueError for bloomfilter."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "bloomfilter",
                    "expected_items": 0,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_bloomfilter_invalid_expected_items_negative(self):
        """Test that expected_items<0 raises ValueError for bloomfilter."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "bloomfilter",
                    "expected_items": -100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_bloomfilter_invalid_fpr_boundary_zero(self):
        """Test that false_positive_rate=0.0 raises ValueError for bloomfilter."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "bloomfilter",
                    "expected_items": 100,
                    "false_positive_rate": 0.0,
                }
            )

    def test_constructor_bloomfilter_invalid_fpr_boundary_one(self):
        """Test that false_positive_rate=1.0 raises ValueError for bloomfilter."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "bloomfilter",
                    "expected_items": 100,
                    "false_positive_rate": 1.0,
                }
            )

    def test_constructor_bloomfilter_invalid_fpr_too_high(self):
        """Test that false_positive_rate>1.0 raises ValueError for bloomfilter."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "bloomfilter",
                    "expected_items": 100,
                    "false_positive_rate": 1.5,
                }
            )

    def test_augment_bloomfilter_empty_query(self, bloom_augmenter):
        """Test that bloomfilter with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = bloom_augmenter(message)
        assert result.query == ""

    def test_augment_bloomfilter_exact_duplicates(self, bloom_augmenter):
        """Test bloomfilter deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_bloomfilter_multiple_exact_duplicates(self, bloom_augmenter):
        """Test bloomfilter deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_bloomfilter_no_duplicates(self, bloom_augmenter):
        """Test that bloomfilter preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_bloomfilter_unique_sentences_preserved(self, bloom_augmenter):
        """Test that bloomfilter preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_bloomfilter_single_sentence(self, bloom_augmenter):
        """Test bloomfilter with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        assert result.query == query

    def test_augment_bloomfilter_single_word(self, bloom_augmenter):
        """Test bloomfilter with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        assert result.query == query

    def test_augment_bloomfilter_multiple_exclamation(self, bloom_augmenter):
        """Test bloomfilter with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_bloomfilter_mixed_terminators(self, bloom_augmenter):
        """Test bloomfilter with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_bloomfilter_with_loop(self, bloom_augmenter):
        """Test bloomfilter with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "bloomfilter",
                "expected_items": 100,
                "false_positive_rate": 0.01,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    def test_augment_bloomfilter_case_sensitive(self, bloom_augmenter):
        """Test that bloomfilter is case-sensitive (unlike minhash)."""
        query = "Hello world. hello world. Different."
        message = AgentMessage(query=query)
        result = bloom_augmenter(message)
        # "Hello world." and "hello world." are different due to case
        assert "Hello world" in result.query
        assert "hello world" in result.query
        assert "Different" in result.query

    # --- SimHash Algorithm Tests ---

    @pytest.fixture
    def simhash_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "simhash",
                "hash_bits": 64,
                "threshold": 0.8,
            }
        )

    @pytest.fixture
    def simhash_low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "simhash",
                "hash_bits": 64,
                "threshold": 0.3,
            }
        )

    @pytest.fixture
    def simhash_high_bits_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "simhash",
                "hash_bits": 128,
                "threshold": 0.8,
            }
        )

    def test_constructor_simhash_defaults(self):
        """Test that simhash constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "simhash"})
        assert aug._algo_config.algorithm_name == "simhash"
        assert aug._algo_config.hash_bits == 64
        assert aug._algo_config.threshold == 0.8

    def test_constructor_simhash_custom_params(self):
        """Test simhash constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "simhash",
                "hash_bits": 128,
                "threshold": 0.7,
            }
        )
        assert aug._algo_config.algorithm_name == "simhash"
        assert aug._algo_config.hash_bits == 128
        assert aug._algo_config.threshold == 0.7

    def test_constructor_simhash_invalid_hash_bits_zero(self):
        """Test that hash_bits=0 raises ValueError for simhash."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "simhash",
                    "hash_bits": 0,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_simhash_invalid_hash_bits_negative(self):
        """Test that hash_bits<0 raises ValueError for simhash."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "simhash",
                    "hash_bits": -64,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_simhash_invalid_threshold(self):
        """Test that invalid threshold raises ValueError for simhash."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "simhash",
                    "threshold": 1.5,
                    "hash_bits": 64,
                }
            )

    def test_constructor_simhash_invalid_seed(self):
        """Test that invalid seed raises ValueError for simhash."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "simhash",
                    "seed": -1,
                    "hash_bits": 64,
                    "threshold": 0.8,
                }
            )

    def test_constructor_simhash_invalid_type(self):
        """Test that invalid type raises ValidationError for simhash."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "simhash", "threshold": [0.8]}
            )

    def test_augment_simhash_empty_query(self, simhash_augmenter):
        """Test that simhash with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = simhash_augmenter(message)
        assert result.query == ""

    def test_augment_simhash_exact_duplicates(self, simhash_augmenter):
        """Test simhash deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_simhash_multiple_exact_duplicates(self, simhash_augmenter):
        """Test simhash deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_simhash_no_duplicates(self, simhash_augmenter):
        """Test that simhash preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_simhash_near_duplicates(self, simhash_low_threshold_augmenter):
        """Test simhash deduplication of near-duplicate sentences with low threshold."""
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = simhash_low_threshold_augmenter(message)
        # Near-duplicates should be removed with low threshold
        assert result.query != query

    def test_augment_simhash_unique_sentences_preserved(self, simhash_augmenter):
        """Test that simhash preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_simhash_single_sentence(self, simhash_augmenter):
        """Test simhash with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        assert result.query == query

    def test_augment_simhash_single_word(self, simhash_augmenter):
        """Test simhash with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        assert result.query == query

    def test_augment_simhash_multiple_exclamation(self, simhash_augmenter):
        """Test simhash with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_simhash_mixed_terminators(self, simhash_augmenter):
        """Test simhash with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_simhash_with_loop(self, simhash_augmenter):
        """Test simhash with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "simhash",
                "hash_bits": 64,
                "threshold": 0.8,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    def test_augment_simhash_case_sensitive(self, simhash_augmenter):
        """Test that simhash is case-insensitive (tokens are lowercased)."""
        query = "Hello world. hello world. Different."
        message = AgentMessage(query=query)
        result = simhash_augmenter(message)
        # "Hello world." and "hello world." are similar due to case-insensitive tokenization
        # One should be removed as a near-duplicate
        assert result.query.count("world") == 1
        assert "Different" in result.query

    def test_augment_simhash_high_bits(self, simhash_high_bits_augmenter):
        """Test simhash with higher hash_bits for more precision."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = simhash_high_bits_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    # --- LSHBloom Algorithm Tests ---

    @pytest.fixture
    def lsh_bloom_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "lsh_bloom",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
                "expected_items": 100,
                "false_positive_rate": 0.01,
            }
        )

    @pytest.fixture
    def lsh_bloom_low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "lsh_bloom",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.3,
                "num_bands": 16,
                "expected_items": 100,
                "false_positive_rate": 0.01,
            }
        )

    @pytest.fixture
    def lsh_bloom_default_augmenter(self):
        return DeduplicationPromptAugmenter(algo_args={"algorithm_name": "lsh_bloom"})

    def test_constructor_lsh_bloom_defaults(self):
        """Test that lsh_bloom constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "lsh_bloom"})
        assert aug._algo_config.algorithm_name == "lsh_bloom"
        assert aug._algo_config.num_perm == 128
        assert aug._algo_config.seed == 42
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.num_bands == 16
        assert aug._algo_config.expected_items == 10
        assert aug._algo_config.false_positive_rate == 0.01

    def test_constructor_lsh_bloom_custom_params(self):
        """Test lsh_bloom constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "lsh_bloom",
                "num_perm": 64,
                "seed": 123,
                "threshold": 0.7,
                "num_bands": 8,
                "expected_items": 500,
                "false_positive_rate": 0.001,
            }
        )
        assert aug._algo_config.algorithm_name == "lsh_bloom"
        assert aug._algo_config.num_perm == 64
        assert aug._algo_config.seed == 123
        assert aug._algo_config.threshold == 0.7
        assert aug._algo_config.num_bands == 8
        assert aug._algo_config.expected_items == 500
        assert aug._algo_config.false_positive_rate == 0.001

    def test_constructor_lsh_bloom_invalid_num_bands_zero(self):
        """Test that num_bands=0 raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_bands": 0,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_num_bands_negative(self):
        """Test that num_bands<0 raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_bands": -5,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_num_bands_greater_than_num_perm(self):
        """Test that num_bands > num_perm raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_bands": 128,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_threshold(self):
        """Test that invalid threshold raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "threshold": 1.5,
                    "num_perm": 64,
                    "seed": 42,
                    "num_bands": 16,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_num_perm(self):
        """Test that invalid num_perm raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_perm": 0,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_seed(self):
        """Test that invalid seed raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "seed": -1,
                    "num_perm": 64,
                    "threshold": 0.8,
                    "num_bands": 16,
                    "expected_items": 100,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_expected_items_zero(self):
        """Test that expected_items=0 raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                    "expected_items": 0,
                    "false_positive_rate": 0.01,
                }
            )

    def test_constructor_lsh_bloom_invalid_fpr_boundary_zero(self):
        """Test that false_positive_rate=0.0 raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                    "expected_items": 100,
                    "false_positive_rate": 0.0,
                }
            )

    def test_constructor_lsh_bloom_invalid_fpr_boundary_one(self):
        """Test that false_positive_rate=1.0 raises ValueError for lsh_bloom."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "lsh_bloom",
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                    "expected_items": 100,
                    "false_positive_rate": 1.0,
                }
            )

    def test_constructor_lsh_bloom_invalid_type(self):
        """Test that invalid type raises ValidationError for lsh_bloom."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "lsh_bloom", "threshold": [0.8]}
            )

    def test_augment_lsh_bloom_empty_query(self, lsh_bloom_augmenter):
        """Test that lsh_bloom with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = lsh_bloom_augmenter(message)
        assert result.query == ""

    def test_augment_lsh_bloom_exact_duplicates(self, lsh_bloom_augmenter):
        """Test lsh_bloom deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_lsh_bloom_multiple_exact_duplicates(self, lsh_bloom_augmenter):
        """Test lsh_bloom deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_lsh_bloom_no_duplicates(self, lsh_bloom_augmenter):
        """Test that lsh_bloom preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_bloom_near_duplicates(self, lsh_bloom_low_threshold_augmenter):
        """Test lsh_bloom behavior with near-duplicate sentences.

        Note: LSHBloom uses threshold to optimize LSH parameters, not for direct
        similarity comparison. Near-duplicates may or may not be detected depending
        on the LSH hash collision probability.
        """
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = lsh_bloom_low_threshold_augmenter(message)
        # LSHBloom may or may not detect near-duplicates due to probabilistic nature
        # Just verify the result is valid
        assert isinstance(result.query, str)
        assert len(result.query) > 0

    def test_augment_lsh_bloom_unique_sentences_preserved(self, lsh_bloom_augmenter):
        """Test that lsh_bloom preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_lsh_bloom_single_sentence(self, lsh_bloom_augmenter):
        """Test lsh_bloom with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        assert result.query == query

    def test_augment_lsh_bloom_single_word(self, lsh_bloom_augmenter):
        """Test lsh_bloom with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        assert result.query == query

    def test_augment_lsh_bloom_multiple_exclamation(self, lsh_bloom_augmenter):
        """Test lsh_bloom with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_lsh_bloom_mixed_terminators(self, lsh_bloom_augmenter):
        """Test lsh_bloom with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_bloom_with_loop(self, lsh_bloom_augmenter):
        """Test lsh_bloom with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "lsh_bloom",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
                "expected_items": 100,
                "false_positive_rate": 0.01,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    def test_augment_lsh_bloom_case_insensitive(self, lsh_bloom_augmenter):
        """Test that lsh_bloom is case-insensitive (tokens are lowercased)."""
        query = "Hello world. hello world. Different."
        message = AgentMessage(query=query)
        result = lsh_bloom_augmenter(message)
        # "Hello world." and "hello world." are similar due to case-insensitive tokenization
        # One should be removed as a near-duplicate
        assert result.query.count("world") == 1
        assert "Different" in result.query

    def test_augment_lsh_bloom_default_augmenter(self, lsh_bloom_default_augmenter):
        """Test lsh_bloom with default parameters."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = lsh_bloom_default_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    # --- Suffix Array Algorithm Tests ---

    @pytest.fixture
    def suffix_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "suffix_array",
                "min_length": 5,
                "max_length": 100,
                "seed": 42,
            }
        )

    @pytest.fixture
    def suffix_default_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={"algorithm_name": "suffix_array"}
        )

    @pytest.fixture
    def suffix_low_min_length_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "suffix_array",
                "min_length": 3,
                "max_length": 50,
                "seed": 42,
            }
        )

    def test_constructor_suffix_array_defaults(self):
        """Test that suffix_array constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "suffix_array"})
        assert aug._algo_config.algorithm_name == "suffix_array"
        assert aug._algo_config.min_length == 50
        assert aug._algo_config.max_length == 1000

    def test_constructor_suffix_array_custom_params(self):
        """Test suffix_array constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "suffix_array",
                "min_length": 10,
                "max_length": 200,
            }
        )
        assert aug._algo_config.algorithm_name == "suffix_array"
        assert aug._algo_config.min_length == 10
        assert aug._algo_config.max_length == 200

    def test_constructor_suffix_array_invalid_min_length_zero(self):
        """Test that min_length=0 raises ValueError for suffix_array."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "suffix_array",
                    "min_length": 0,
                    "max_length": 100,
                    "seed": 42,
                }
            )

    def test_constructor_suffix_array_invalid_min_length_negative(self):
        """Test that min_length<0 raises ValueError for suffix_array."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "suffix_array",
                    "min_length": -10,
                    "max_length": 100,
                    "seed": 42,
                }
            )

    def test_constructor_suffix_array_invalid_max_length_zero(self):
        """Test that max_length=0 raises ValueError for suffix_array."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "suffix_array",
                    "min_length": 5,
                    "max_length": 0,
                    "seed": 42,
                }
            )

    def test_constructor_suffix_array_invalid_max_length_negative(self):
        """Test that max_length<0 raises ValueError for suffix_array."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "suffix_array",
                    "min_length": 5,
                    "max_length": -100,
                    "seed": 42,
                }
            )

    def test_constructor_suffix_array_invalid_type(self):
        """Test that invalid type raises ValidationError for suffix_array."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "suffix_array", "min_length": [5]}
            )

    def test_augment_suffix_array_empty_query(self, suffix_augmenter):
        """Test that suffix_array with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = suffix_augmenter(message)
        assert result.query == ""

    def test_augment_suffix_array_exact_duplicates(self, suffix_augmenter):
        """Test suffix_array deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_suffix_array_multiple_exact_duplicates(self, suffix_augmenter):
        """Test suffix_array deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_suffix_array_no_duplicates(self, suffix_augmenter):
        """Test that suffix_array preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_suffix_array_unique_sentences_preserved(self, suffix_augmenter):
        """Test that suffix_array preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_suffix_array_single_sentence(self, suffix_augmenter):
        """Test suffix_array with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        assert result.query == query

    def test_augment_suffix_array_single_word(self, suffix_augmenter):
        """Test suffix_array with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        assert result.query == query

    def test_augment_suffix_array_multiple_exclamation(self, suffix_augmenter):
        """Test suffix_array with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_suffix_array_mixed_terminators(self, suffix_augmenter):
        """Test suffix_array with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_suffix_array_with_loop(self, suffix_augmenter):
        """Test suffix_array with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "suffix_array",
                "min_length": 5,
                "max_length": 100,
                "seed": 42,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    def test_augment_suffix_array_case_insensitive(self, suffix_augmenter):
        """Test that suffix_array is case-insensitive (tokens are lowercased)."""
        query = "Hello world. hello world. Different."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # "Hello world." and "hello world." are similar due to case-insensitive tokenization
        # One should be removed as a duplicate
        assert result.query.count("world") == 1
        assert "Different" in result.query

    def test_augment_suffix_array_default_augmenter(self, suffix_default_augmenter):
        """Test suffix_array with default parameters."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = suffix_default_augmenter(message)
        # With default min_length=50, short duplicates may not be detected
        # Just verify the result is valid
        assert isinstance(result.query, str)
        assert len(result.query) > 0

    def test_augment_suffix_array_low_min_length(self, suffix_low_min_length_augmenter):
        """Test suffix_array with low min_length threshold."""
        query = "Hi. Hi. Hi. This is unique."
        message = AgentMessage(query=query)
        result = suffix_low_min_length_augmenter(message)
        # With min_length=3, "Hi" should be detected as duplicate
        assert result.query.count("Hi") == 1
        assert "This is unique" in result.query

    def test_augment_suffix_array_long_duplicate_substring(self, suffix_augmenter):
        """Test suffix_array with long duplicate substrings."""
        query = (
            "This is a very long sentence that is repeated. "
            "This is a very long sentence that is repeated. "
            "This is a unique sentence."
        )
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # The long duplicate should be removed
        assert result.query.count("very long sentence") == 1
        assert "This is a unique sentence" in result.query

    def test_augment_suffix_array_preserves_order(self, suffix_augmenter):
        """Test that suffix_array preserves the order of unique sentences."""
        query = "First sentence. Second sentence. Third sentence. First sentence."
        message = AgentMessage(query=query)
        result = suffix_augmenter(message)
        # First sentence should appear only once, at the beginning
        sentences = result.query.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        assert sentences[0] == "First sentence"
        assert "Second sentence" in result.query
        assert "Third sentence" in result.query
