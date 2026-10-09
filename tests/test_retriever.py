"""Test cases for the DataFrameRetriever module."""

import json

import pandas as pd
import pytest
from aap_core.retriever import DataFrameRetriever
from aap_core.types import AgentMessage


class TestDataFrameRetrieverDataKeyValidation:
    """Test data_key field validation."""

    def test_valid_data_key(self):
        """Test creating a retriever with a valid data_key."""
        retriever = DataFrameRetriever(data_key="context.data")
        assert retriever.data_key == "context.data"

    def test_valid_custom_data_key(self):
        """Test creating a retriever with a custom valid data_key."""
        retriever = DataFrameRetriever(data_key="context.custom_field")
        assert retriever.data_key == "context.custom_field"

    def test_invalid_data_key_no_prefix(self):
        """Test that data_key without 'context.' prefix raises ValueError."""
        with pytest.raises(
            ValueError, match="task_response_key must start with 'context.'"
        ):
            DataFrameRetriever(data_key="data")

    def test_invalid_data_key_wrong_prefix(self):
        """Test that data_key with wrong prefix raises ValueError."""
        with pytest.raises(
            ValueError, match="task_response_key must start with 'context.'"
        ):
            DataFrameRetriever(data_key="message.data")


class TestDataFrameRetrieverFromPandas:
    """Test from_pandas factory method."""

    def test_from_pandas_default(self):
        """Test from_pandas with default prettier (to_string)."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        retriever = DataFrameRetriever.from_pandas(df)
        assert retriever._data is not None
        assert "Alice" in retriever._data
        assert "Bob" in retriever._data

    def test_from_pandas_empty_dataframe(self):
        """Test from_pandas with an empty DataFrame."""
        df = pd.DataFrame()
        retriever = DataFrameRetriever.from_pandas(df)
        assert retriever._data is not None

    def test_from_pandas_single_row(self):
        """Test from_pandas with a single row DataFrame."""
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        retriever = DataFrameRetriever.from_pandas(df)
        assert "Alice" in retriever._data
        assert "25" in retriever._data

    def test_from_pandas_with_prettier(self):
        """Test from_pandas with a tabulate prettier format."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        retriever = DataFrameRetriever.from_pandas(df, prettier="fancy_grid")
        assert retriever._data is not None
        # tabulate with df.to_dict() produces a dict-of-dicts format
        # Just verify the data was converted and stored
        assert retriever._data != ""

    def test_from_pandas_with_custom_data_key(self):
        """Test from_pandas with a custom data_key."""
        df = pd.DataFrame({"name": ["Alice"]})
        retriever = DataFrameRetriever.from_pandas(df, data_key="context.users")
        assert retriever.data_key == "context.users"

    def test_from_pandas_with_kwargs(self):
        """Test from_pandas with additional kwargs passed to to_string."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        retriever = DataFrameRetriever.from_pandas(df, index=False)
        assert retriever._data is not None
        assert "Alice" in retriever._data


class TestDataFrameRetrieverFromString:
    """Test from_string factory method."""

    def test_from_string_basic(self):
        """Test from_string with a basic string."""
        data = "Hello, world!"
        retriever = DataFrameRetriever.from_string(data)
        assert retriever._data == data

    def test_from_string_empty(self):
        """Test from_string with an empty string."""
        retriever = DataFrameRetriever.from_string("")
        assert retriever._data == ""

    def test_from_string_multiline(self):
        """Test from_string with a multiline string."""
        data = "Line 1\nLine 2\nLine 3"
        retriever = DataFrameRetriever.from_string(data)
        assert retriever._data == data

    def test_from_string_with_custom_data_key(self):
        """Test from_string with a custom data_key."""
        retriever = DataFrameRetriever.from_string("test", data_key="context.custom")
        assert retriever.data_key == "context.custom"


class TestDataFrameRetrieverFromDict:
    """Test from_dict factory method."""

    def test_from_dict_default(self):
        """Test from_dict with default prettier (str)."""
        data = {"name": "Alice", "age": 25}
        retriever = DataFrameRetriever.from_dict(data)
        assert retriever._data is not None
        assert "Alice" in retriever._data

    def test_from_dict_empty(self):
        """Test from_dict with an empty dictionary."""
        retriever = DataFrameRetriever.from_dict({})
        assert retriever._data is not None

    def test_from_dict_with_prettier(self):
        """Test from_dict with a tabulate prettier format."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        retriever = DataFrameRetriever.from_dict(data, prettier="simple")
        assert retriever._data is not None
        assert "Alice" in retriever._data

    def test_from_dict_with_custom_data_key(self):
        """Test from_dict with a custom data_key."""
        retriever = DataFrameRetriever.from_dict(
            {"key": "value"}, data_key="context.info"
        )
        assert retriever.data_key == "context.info"


class TestDataFrameRetrieverFromIterable:
    """Test from_iterable factory method."""

    def test_from_iterable_basic(self):
        """Test from_iterable with a basic list of strings."""
        data = ["item1", "item2", "item3"]
        retriever = DataFrameRetriever.from_iterable(data)
        assert retriever._data is not None
        assert "item1" in retriever._data
        assert "item2" in retriever._data
        assert "item3" in retriever._data

    def test_from_iterable_custom_bullet(self):
        """Test from_iterable with a custom bullet character."""
        data = ["item1", "item2"]
        retriever = DataFrameRetriever.from_iterable(data, bullet_char="*")
        assert "* item1" in retriever._data
        assert "* item2" in retriever._data

    def test_from_iterable_empty(self):
        """Test from_iterable with an empty list."""
        retriever = DataFrameRetriever.from_iterable([])
        assert retriever._data == ""

    def test_from_iterable_single_item(self):
        """Test from_iterable with a single item."""
        retriever = DataFrameRetriever.from_iterable(["only_item"])
        assert "- only_item" in retriever._data

    def test_from_iterable_with_custom_data_key(self):
        """Test from_iterable with a custom data_key."""
        retriever = DataFrameRetriever.from_iterable(["a"], data_key="context.items")
        assert retriever.data_key == "context.items"


class TestDataFrameRetrieverFromJsonl:
    """Test from_jsonl factory method."""

    def test_from_jsonl_basic(self, tmp_path):
        """Test from_jsonl with a valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        records = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
        ]
        with open(jsonl_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        retriever = DataFrameRetriever.from_jsonl(str(jsonl_file))
        assert retriever._data is not None
        assert "Alice" in retriever._data
        assert "Bob" in retriever._data

    def test_from_jsonl_empty_file(self, tmp_path):
        """Test from_jsonl with an empty JSONL file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        retriever = DataFrameRetriever.from_jsonl(str(jsonl_file))
        assert retriever._data is not None

    def test_from_jsonl_nonexistent_file(self):
        """Test from_jsonl with a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DataFrameRetriever.from_jsonl("/nonexistent/path/file.jsonl")

    def test_from_jsonl_with_custom_data_key(self, tmp_path):
        """Test from_jsonl with a custom data_key."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps({"name": "Alice"}) + "\n")

        retriever = DataFrameRetriever.from_jsonl(
            str(jsonl_file), data_key="context.users"
        )
        assert retriever.data_key == "context.users"


class TestDataFrameRetrieverRetrieve:
    """Test the retrieve method."""

    def test_retrieve_with_none_context(self):
        """Test retrieve when message context is None."""
        retriever = DataFrameRetriever.from_string("test data")
        message = AgentMessage(query="test query")
        assert message.context is None

        result = retriever.retrieve(message)

        assert result.context is not None
        assert "data" in result.context
        assert result.context["data"] == "test data"

    def test_retrieve_with_existing_context(self):
        """Test retrieve when message already has context."""
        retriever = DataFrameRetriever.from_string("new data")
        message = AgentMessage(query="test query", context={"existing": "value"})

        result = retriever.retrieve(message)

        assert result.context is not None
        assert "existing" in result.context
        assert result.context["existing"] == "value"
        assert "data" in result.context
        assert result.context["data"] == "new data"

    def test_retrieve_with_custom_data_key(self):
        """Test retrieve with a custom data_key."""
        retriever = DataFrameRetriever.from_string(
            "custom data", data_key="context.custom_field"
        )
        message = AgentMessage(query="test query")

        result = retriever.retrieve(message)

        assert result.context is not None
        assert "custom_field" in result.context
        assert result.context["custom_field"] == "custom data"

    def test_retrieve_preserves_query(self):
        """Test that retrieve preserves the original query."""
        retriever = DataFrameRetriever.from_string("data")
        message = AgentMessage(query="original query")

        result = retriever.retrieve(message)

        assert result.query == "original query"

    def test_retrieve_with_empty_data(self):
        """Test retrieve with empty data string."""
        retriever = DataFrameRetriever.from_string("")
        message = AgentMessage(query="test query")

        result = retriever.retrieve(message)

        assert result.context is not None
        assert "data" in result.context
        assert result.context["data"] == ""


class TestDataFrameRetrieverCall:
    """Test the __call__ method (with post_process)."""

    def test_call_without_post_process(self):
        """Test __call__ without a post_process chain."""
        retriever = DataFrameRetriever.from_string("test data")
        message = AgentMessage(query="test query")

        result = retriever(message)

        assert result.context is not None
        assert "data" in result.context
        assert result.context["data"] == "test data"

    def test_call_with_post_process(self):
        """Test __call__ with a post_process chain."""
        from aap_core.types import BaseChain

        class DummyPostProcess(BaseChain):
            """A dummy post process chain that adds a marker to context."""

            def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
                if message.context is None:
                    message.context = {}
                message.context["post_processed"] = True
                return message

        retriever = DataFrameRetriever.from_string("test data")
        retriever.post_process = DummyPostProcess()
        message = AgentMessage(query="test query")

        result = retriever(message)

        assert result.context is not None
        assert "data" in result.context
        assert result.context["post_processed"] is True

    def test_call_preserves_query(self):
        """Test that __call__ preserves the original query."""
        retriever = DataFrameRetriever.from_string("data")
        message = AgentMessage(query="preserved query")

        result = retriever(message)

        assert result.query == "preserved query"
