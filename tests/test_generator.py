from types import SimpleNamespace

from rag.generator import NOT_FOUND, answer, build_user_prompt

PASSAGES = [
    {"text": "Canary gets 5% of traffic.", "source": "handbook.md", "page": 0},
    {"text": "Rollback within 15 minutes.", "source": "guide.pdf", "page": 3},
]


class FakeClient:
    def __init__(self, reply="The canary gets 5% [1]."):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=f"  {self.reply}  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_prompt_numbers_sources_and_pages():
    prompt = build_user_prompt("How much traffic?", PASSAGES)
    assert "[1] (handbook.md)" in prompt
    assert "[2] (guide.pdf, page 3)" in prompt
    assert "Question: How much traffic?" in prompt


def test_no_passages_skips_llm_call():
    client = FakeClient()
    assert answer("anything", [], client=client) == NOT_FOUND
    assert client.calls == []


def test_answer_calls_llm_with_grounding_prompt():
    client = FakeClient()
    reply = answer("How much traffic?", PASSAGES, client=client, model="test-model")
    assert reply == "The canary gets 5% [1]."
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0
    assert "ONLY" in call["messages"][0]["content"]
