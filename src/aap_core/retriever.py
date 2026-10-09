import abc
import json
from collections.abc import Iterable
import os

import pandas as pd
import tabulate
from pydantic import Field, PrivateAttr, field_validator

from aap_core.types import AgentMessage, BaseChain


class BaseRetriever(BaseChain):
    """Base class for retriever.
    A retriever is responsible for retrieving data from a source and adding it to the message context.
    The retriever can be used to retrieve data from a database, a file, an API, etc.
    The retrieved data will be added to the message context and can be used by the agent"""

    post_process: BaseChain | None = Field(
        default=None, description="The post process function if any"
    )

    @abc.abstractmethod
    def retrieve(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.retrieve(message, **kwargs)
        if self.post_process:
            # reranker, summarizer, etc
            # The implementation is stay entirely in the post_process object
            message = self.post_process(message)
        return message


class DataFrameRetriever(BaseRetriever):
    """
    A class that retrieves data from a pandas DataFrame.

    This class is used to retrieve data from a pandas DataFrame. It takes in a pandas DataFrame as input and returns the data as a string.
    """

    _data: str | None = PrivateAttr(None)
    data_key: str = Field(
        default="context.data", description="The key to the data in the message"
    )

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("task_response_key must start with 'context.'")
        return v

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        data_key: str = "context.data",
        prettier: str | tabulate.TableFormat | None = None,
        **kwargs,
    ):
        """
        Parse a pandas dataframe into a string format.
        Tip: to parse file such as csv, parquet, etc. we can use this factory method to create a retriever

        Args:
            data (pd.DataFrame): The pandas dataframe to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.
            [toon](https://github.com/toon-format/spec) are also supported

            **kwargs: Additional keyword arguments to be passed to the conversion statement.
            - For pandas.DataFrame.to_string(), see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
            - For tabulate.tabulate(), see https://pypi.org/project/tabulate/

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is 'toon', the dataframe is converted to a string following toon specification
            - If prettier is None, the dataframe is converted to a string using the to_string() method.
            - If prettier is not None, the dataframe is converted to a string using the tabulate module with the specified tablefmt.
        """
        retriever = DataFrameRetriever(data_key=data_key)
        if prettier is None:
            retriever._data = data.to_string(**kwargs)
        # elif prettier == "toon":      # toon_format currently is beta release, waiting 1.0.0 release
        #     retriever._data = toon_format.encode(data.to_dict())
        else:
            retriever._data = tabulate.tabulate(
                data.to_dict(), tablefmt=prettier, **kwargs
            )

        return retriever

    @classmethod
    def from_string(cls, data: str, data_key: str = "context.data"):
        """
        Parse a string into a string.

        Args:
            data (str): The string to be parsed.

        Returns:
            self: The instance of the class.
        """
        retriever = DataFrameRetriever(data_key=data_key)
        retriever._data = data
        return retriever

    @classmethod
    def from_dict(
        cls,
        data: dict,
        data_key: str = "context.data",
        prettier: str | tabulate.TableFormat | None = None,
        **kwargs,
    ):
        """
        Parse a dictionary into a string.

        Args:
            data (dict): The dictionary to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.
            [toon](https://github.com/toon-format/spec) are also supported

            **kwargs: Additional keyword arguments to be passed to the conversion statement.
            - For tabulate.tabulate(), see https://pypi.org/project/tabulate/

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is 'toon', the dataframe is converted to a string following toon specification
            - If prettier is None, the dictionary is converted to a string using the str() method.
            - If prettier is not None, the dictionary is converted to a string using the tabulate module with the specified tablefmt.
        """
        retriever = DataFrameRetriever(data_key=data_key)
        if prettier is None:
            retriever._data = str(data)
        # elif prettier == "toon":      # toon_format currently is beta release, waiting 1.0.0 release
        #     retriever._data = toon_format.encode(data.to_dict())
        else:
            retriever._data = tabulate.tabulate(data, tablefmt=prettier, **kwargs)

        return retriever

    @classmethod
    def from_iterable(
        cls, data: Iterable[str], data_key: str = "context.data", bullet_char: str = "-"
    ):
        """
        Parse a list of strings into a string.

        Args:
            data (List[str]): The list of strings to be parsed.
            bullet_char (str, optional): The character to be used as the bullet. Defaults to "-".

        Returns:
            self: The instance of the class.

        Notes:
            - The list of strings is converted to a string using the join() method with the bullet character as the separator.
        """
        retriever = DataFrameRetriever(data_key=data_key)
        retriever._data = "\n".join([f"{bullet_char} {d}" for d in data])
        return retriever

    @classmethod
    def from_jsonl(
        cls, path: str | os.PathLike[str], data_key: str = "context.data", **kwargs
    ):
        """
        Parse a JSONL file into a string.

        This method reads a JSONL file line by line and converts each line to a dictionary using the json.loads() method.
        The resulting dictionary is then converted to a pandas DataFrame before initiate the object

        Args:
            path (str): The path to the JSONL file.
            **kwargs: Additional keyword arguments to be passed to the DataFrameRetriever.from_pandas() method.

        Returns:
            self: The instance of the class.

        Raises:
            FileNotFoundError: If the file specified by path does not exist.

        Notes:
            - The JSONL file is parsed line by line using the json.loads() method.
            - The parsed data is converted to a pandas DataFrame and passed to the parse_pandas() method.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")

        # Implementation provided by https://stackoverflow.com/a/74406107
        with open(path, "r") as f:
            lines = f.read().splitlines()

        line_dicts = [json.loads(line) for line in lines]
        return DataFrameRetriever.from_pandas(
            pd.DataFrame(line_dicts), data_key=data_key, **kwargs
        )

    def retrieve(self, message: AgentMessage, **kwargs) -> AgentMessage:
        context_key = self.data_key.replace("context.", "")
        if message.context is None:
            message.context = {context_key: self._data}
        else:
            message.context[context_key] = self._data
        return message
