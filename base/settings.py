"""Service settings."""

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

from util.logger import configure_logger

load_dotenv(find_dotenv())
logger = configure_logger(__name__)


class Settings(BaseSettings):
    google_url: str = "https://www.googleapis.com/customsearch/v1"
    api_key: str
    cx_code: str
    model: str


SETTINGS = Settings()

logger.info("loaded settings: {settings}".format(settings=SETTINGS.model_dump()))
