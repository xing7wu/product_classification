from pydantic import BaseModel


class Title(BaseModel):
    name: str | list


class Category(BaseModel):
    name: str | list
