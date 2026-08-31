from moatless.agent.code_agent import create_base_actions
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.repository.repository import InMemRepository


def test_base_actions_without_code_index_use_repository_native_search():
    actions = create_base_actions(
        repository=InMemRepository(files={"example.py": "def example():\n    pass\n"}),
        code_index=None,
        completion_model=CompletionModel(
            model="test-model",
            response_format=LLMResponseFormat.TOOLS,
        ),
    )

    action_names = [action.name for action in actions]

    assert action_names == ["ListFiles", "FindCodeSnippet", "ViewCode"]
    assert "SemanticSearch" not in action_names
    assert "FindClass" not in action_names
    assert "FindFunction" not in action_names
