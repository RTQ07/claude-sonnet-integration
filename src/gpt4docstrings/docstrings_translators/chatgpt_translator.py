import ast
import os
import textwrap
from anthropic import Anthropic

from gpt4docstrings.docstring import Docstring
from gpt4docstrings.docstrings_translators.base import DocstringTranslator
from gpt4docstrings.prompts.translation.chatgpt import PROMPT
from gpt4docstrings.utils.decorators import retry
from gpt4docstrings.utils.parsers import DocstringParser
from gpt4docstrings.visit import GPT4DocstringsNode


class ChatGPTDocstringTranslator(DocstringTranslator):
    """A class for generating Python docstrings using Claude."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        docstring_style: str,
    ):
        self.api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Please, provide the Anthropic API Key")

        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.docstring_style = docstring_style
        self.prompt_template = PROMPT

    async def _get_completion(self, prompt: str) -> str:
        """
        Generates a completion using the Claude model.

        Args:
            prompt (str): The prompt for generating the completion.

        Returns:
            str: The generated completion.
        """
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    @retry()
    async def translate_docstring(self, node: GPT4DocstringsNode) -> Docstring:
        """
        Translates a docstring for a function.

        Args:
            node (GPT4DocstringsNode): A GPT4DocstringsNode node

        Returns:
            Docstring: A Docstring object
        """
        docstring = ast.get_docstring(node.ast_node)
        stripped_source = textwrap.dedent(docstring)
        parent_offset = node.col_offset

        prompt = self.prompt_template.format(
            docstring=stripped_source, style=self.docstring_style
        )
        translated_docstring = await self._get_completion(prompt)
        docstring = DocstringParser().parse(translated_docstring)

        return Docstring(
            text=docstring, col_offset=4 + parent_offset, lineno=node.docstring_lineno
        )