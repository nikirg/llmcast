from llmcast import BaseTemplate


class GreetTemplate(BaseTemplate):
    """Hello, {{ name }}! Format: {{ output_format }}."""

    name: str


class EmptyTemplate(BaseTemplate):
    pass


def test_renders_variables():
    tmpl = GreetTemplate(name="World")
    assert str(tmpl) == "Hello, World! Format: json."


def test_default_output_format():
    tmpl = GreetTemplate(name="x")
    assert tmpl.output_format == "json"


def test_custom_output_format():
    tmpl = GreetTemplate(name="x", output_format="yaml")
    assert tmpl.output_format == "yaml"
    assert "yaml" in str(tmpl)


def test_empty_docstring_returns_empty():
    tmpl = EmptyTemplate()
    assert str(tmpl) == ""


def test_jinja2_conditional():
    class ConditionalTemplate(BaseTemplate):
        """{% if flag %}yes{% else %}no{% endif %}"""

        flag: bool

    assert str(ConditionalTemplate(flag=True)) == "yes"
    assert str(ConditionalTemplate(flag=False)) == "no"


def test_jinja2_loop():
    class ListTemplate(BaseTemplate):
        """{% for item in items %}{{ item }}{% endfor %}"""

        items: list[str]

    assert str(ListTemplate(items=["a", "b", "c"])) == "abc"
