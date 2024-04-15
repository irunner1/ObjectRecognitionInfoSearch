import requests

from base.settings import SETTINGS
from utils.logger import configure_logger

logger = configure_logger(__name__)


def search_info(object_name: str) -> dict | None:
    """Search data with google api.

    Args:
        object_name: class name of recognized object

    Returns:
        dict | None: founded data
    """

    api_key = SETTINGS.api_key
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": SETTINGS.cx_code,
        "q": object_name,
    }

    try:
        response = requests.get(url=url, params=params)
    except Exception as error:
        logger.warning(
            "ERROR {error} with {data}; {url}".format(
                error=error, data=object_name, url=url
            )
        )
        return None

    data = response.json()

    return data["items"]
