import os
import textwrap
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from gpt4docstrings.docstring import Docstring
from gpt4docstrings.docstrings_generators.base import DocstringGenerator
from gpt4docstrings.prompts.generation.chatgpt import CLASS_PROMPTS, FUNCTION_PROMPTS
from gpt4docstrings.utils.decorators import retry
from gpt4docstrings.utils.parsers import DocstringParser
from gpt4docstrings.visit import GPT4DocstringsNode

class ChatGPTDocstringGenerator(DocstringGenerator):
    """A class for generating Python docstrings using Claude Sonnet."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        docstring_style: str,
    ):
        self.api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide the Anthropic API Key")
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.docstring_style = docstring_style
        self.function_prompt_template = FUNCTION_PROMPTS.get(docstring_style)
        self.class_prompt_template = CLASS_PROMPTS.get(docstring_style)

    async def _get_completion(self, prompt: str) -> str:
        """
        Generates a completion using the Claude Sonnet model.
        Args:
            prompt (str): The prompt for generating the completion.
        Returns:
            str: The generated completion.
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            max_tokens_to_sample=300,
        )
        return response.completion

    def _get_template(self, node: GPT4DocstringsNode):
        """Returns a function template or a class template depending on the node type"""
        if node.node_type in ["FunctionDef", "AsyncFunctionDef"]:
            return self.function_prompt_template
        else:
            return self.class_prompt_template

    @retry()
    async def generate_docstring(self, node: GPT4DocstringsNode) -> Docstring:
        """
        Generates a docstring for a function.
        Args:
            node (GPT4DocstringsNode): A GPT4DocstringsNode node
        Returns:
            Docstring: A Docstring object
        """
        source = node.source.strip()
        stripped_source = textwrap.dedent(source)
        prompt_template = self._get_template(node)
        parent_offset = node.col_offset
        prompt = prompt_template.format(code=stripped_source)
        docstring = DocstringParser().parse(
            await self._get_completion(prompt)
        )
        return Docstring(
            text=docstring, col_offset=4 + parent_offset, lineno=node.docstring_lineno
        )
