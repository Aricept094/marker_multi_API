import json
import time
from io import BytesIO
from typing import List, Annotated

import PIL
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

class BaseGeminiService(BaseService):
    gemini_model_name: Annotated[
        str,
        "The name of the Google model to use for the service."
    ] = "gemini-2.0-flash"

    def img_to_bytes(self, img: PIL.Image.Image):
        image_bytes = BytesIO()
        img.save(image_bytes, format="WEBP")
        return image_bytes.getvalue()

    def get_google_client(self, timeout: int):
        raise NotImplementedError

    def __call__(
            self,
            prompt: str,
            image: PIL.Image.Image | List[PIL.Image.Image],
            block: Block,
            response_schema: type[BaseModel],
            max_retries: int | None = None,
            timeout: int | None = None
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        if not isinstance(image, list):
            image = [image]

        image_parts = [types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/webp") for img in image]

        tries = 0
        while tries < max_retries:
            try:
                client = self.get_google_client(timeout=timeout)
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts + [prompt], # According to gemini docs, it performs better if the image is the first element
                    config={
                        "temperature": 0,
                        "response_schema": response_schema,
                        "response_mime_type": "application/json",
                    },
                )
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                return json.loads(output)
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded
                    tries += 1
                    wait_time = tries * 3
                    print(f"APIError: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(e)
                    break
            except Exception as e:
                print(e)
                break

        return {}


class GoogleGeminiService(BaseGeminiService):
    gemini_api_key: Annotated[
        str,
        "The Google API key to use for the service."
    ] = None
    
    gemini_api_keys: Annotated[
        List[str],
        "A list of Google API keys to use for the service. If one key gets exhausted, the service will switch to the next one."
    ] = None
    
    current_key_index: int = 0
    
    def __init__(self, config: BaseModel | dict = None):
        # Initialize base values
        if config is None:
            config = {}
            
        # Make sure we have at least a default for the API keys
        self.gemini_api_keys = []
        
        # Initialize with base class
        super().__init__(config)
        
        # Handle single key case
        if self.gemini_api_key and not self.gemini_api_keys:
            self.gemini_api_keys = [self.gemini_api_key]
        # Handle API keys list being passed
        elif config and "gemini_api_keys" in config:
            api_keys = config.get("gemini_api_keys", [])
            if api_keys and len(api_keys) > 0:
                self.gemini_api_keys = api_keys
                # Set the first key as the primary key
                self.gemini_api_key = api_keys[0]
        
        # Ensure we have at least one API key
        if not self.gemini_api_keys or len(self.gemini_api_keys) == 0:
            if not self.gemini_api_key:
                raise ValueError("You must provide at least one Gemini API key")
            self.gemini_api_keys = [self.gemini_api_key]
        
        # Make sure primary key is set
        if not self.gemini_api_key and self.gemini_api_keys:
            self.gemini_api_key = self.gemini_api_keys[0]
        
        # Debug output
        print(f"Initialized GoogleGeminiService with {len(self.gemini_api_keys)} API keys")

    def get_google_client(self, timeout: int):
        return genai.Client(
            api_key=self.gemini_api_key,
            http_options={"timeout": timeout * 1000} # Convert to milliseconds
        )
    
    def __call__(
            self,
            prompt: str,
            image: PIL.Image.Image | List[PIL.Image.Image],
            block: Block,
            response_schema: type[BaseModel],
            max_retries: int | None = None,
            timeout: int | None = None
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        if not isinstance(image, list):
            image = [image]

        image_parts = [types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/webp") for img in image]

        # Keep track of which API keys have been tried in this attempt
        tried_keys = set()
        total_tries = 0
        
        while total_tries < max_retries * len(self.gemini_api_keys):
            try:
                # Get current API key
                self.gemini_api_key = self.gemini_api_keys[self.current_key_index]
                tried_keys.add(self.current_key_index)
                
                # Get a client with the current API key
                client = self.get_google_client(timeout=timeout)
                
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts + [prompt], # According to gemini docs, it performs better if the image is the first element
                    config={
                        "temperature": 0,
                        "response_schema": response_schema,
                        "response_mime_type": "application/json",
                    },
                )
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                return json.loads(output)
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded - switch to next API key
                    total_tries += 1
                    
                    # Move to the next API key
                    self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
                    
                    # If we've tried all keys, wait a bit before trying again
                    if len(tried_keys) >= len(self.gemini_api_keys):
                        wait_time = (total_tries // len(self.gemini_api_keys) + 1) * 3
                        print(f"All API keys exhausted. Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"API key exhausted. Switching to next key...")
                else:
                    print(e)
                    break
            except Exception as e:
                print(e)
                break

        return {}
