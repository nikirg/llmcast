from jinja2 import Environment
from pydantic import BaseModel
from textwrap import dedent
from typing import Literal

type ResultFormat = Literal["json", "yaml", "toml"]

env = Environment()


class BaseTemplate(BaseModel):
    output_format: ResultFormat = "json"

    @property
    def template(self) -> str | None:
        if doc := self.__doc__:
            return dedent(doc)

    def __str__(self) -> str:
        template_string = self.template

        if template_string is None:
            raise ValueError("Template string (docstring) not found")

        template = env.from_string(template_string)
        values = self.model_dump(mode="json")

        return template.render(values)
