import asyncio

import openai

from evals.utils import load_secrets

secrets = load_secrets("SECRETS")
try:
    org = secrets["DEFAULT_ORG"]
    openai_api_key = secrets["OPENAI_API_KEY"]
except KeyError:
    print("Organization or API key not found in secrets")
    raise


async def main():
    openai.api_key = openai_api_key
    # If needed, set organization:
    # openai.organization = "org-..."

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.0,
        organization=org,
    )
    print(response)


# Run the async function
asyncio.run(main())
