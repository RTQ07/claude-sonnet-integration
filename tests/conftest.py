import os

import pytest


def pytest_configure():
    pytest.TESTS_PATH = os.path.dirname(__file__)


def generate_test_function_docstring(source: str):
    base_docstring = "This is a generated docstring"
    return {
        "docstring": {"text": f'"""\n{base_docstring}\n"""', "indentation_level": 4}
    }


def generate_test_class_docstring(source: str):
    base_docstring = "This is a generated docstring"
    return {
        "docstring": {"text": f'"""\n{base_docstring}\n"""', "indentation_level": 4},
        "add_word_to_attr1": {
            "text": f'"""\n{base_docstring}\n"""',
            "indentation_level": 8,
        },
        "pow_attr2": {"text": f'"""\n{base_docstring}\n"""', "indentation_level": 8},
    }


@pytest.fixture
def write_source_side_effect():
    def _write_source_side_effect(filename):
        def __write_source_side_effect(_source, _filename):
            _filename = filename
            with open(_filename, "a", encoding="utf-8") as file:
                file.write(_source.dumps())

        return __write_source_side_effect

    return _write_source_side_effect


@pytest.fixture(scope="module")
def test_openai_api_key() -> pytest.fixture():
    os.environ["OPENAI_API_KEY"] = "test_api_key"


@pytest.fixture
def mock_generate_function_docstring(mocker):
    return mocker.patch(
        "gpt4docstrings.generate_docstrings.ChatGPTDocstringGenerator.generate_function_docstring",
        side_effect=generate_test_function_docstring,
    )


@pytest.fixture
def mock_generate_class_docstring(mocker):
    return mocker.patch(
        "gpt4docstrings.generate_docstrings.ChatGPTDocstringGenerator.generate_class_docstring",
        side_effect=generate_test_class_docstring,
    )
