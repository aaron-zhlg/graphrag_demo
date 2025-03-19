import os
import threading

import dotenv
from openai import OpenAI
import json
import re
dotenv.load_dotenv()
import logging


class DeepSeekClient:
    _instance = None
    _lock = threading.Lock()
    _is_initialized = False  # This flag will help to avoid reinitializing

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:  # Double-checked locking pattern
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.deepseek_client = OpenAI(api_key=os.getenv('VOL_DEEPSEEK_API_KEY'),
                                          base_url="https://ark.cn-beijing.volces.com/api/v3")
            self._is_initialized = True


def get_deepseek_client():
    return DeepSeekClient().deepseek_client



def chat_with_deepseek(messages, model='ep-20250222034309-5tnbn', json_format=True):
    if json_format:
        response = get_deepseek_client().chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
    else:
        response = get_deepseek_client().chat.completions.create(
            model=model,
            messages=messages,
        )
    resp = response.choices[0].message.content
    if resp:
        if not json_format:
            return resp
        resp = re.sub(r'`json\n|`', '', resp)
        try:

            keywords_data = json.loads(resp)
            return keywords_data
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON string from vol deepseek: {resp}", )
            return None
    return None