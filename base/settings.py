"""Service settings."""

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

from utils.logger import configure_logger

load_dotenv(find_dotenv())
logger = configure_logger(__name__)


class Settings(BaseSettings):
    api_key: str
    cx_code: str


SETTINGS = Settings()

logger.info("loaded settings: {settings}".format(settings=SETTINGS.model_dump()))
