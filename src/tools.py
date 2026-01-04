from langchain.tools import tool
import aiohttp

@tool
async def validate_fastapi_doc_url(url: str) -> bool:
    """Validate if a FastAPI documentation URL is reachable asynchronously.

    Args:
        url (str): The FastAPI documentation URL to validate.
    Returns:
        bool: True if the URL is reachable (HTTP 200), False otherwise.
    """

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.head(url, allow_redirects=True) as response:
                return response.status == 200
        except aiohttp.ClientError:
            return False