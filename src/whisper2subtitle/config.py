from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    hf_token: str = Field(..., alias="HF_TOKEN")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
