import yaml
import os
from typing import Union, Dict, Optional
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI


def load_config(config_path: str) -> Dict:
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"CRITICAL: Config file not found at '{config_path}'")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict) or "model_info" not in data:
                raise ValueError("Config file is missing 'model_info' section")
            return data["model_info"]
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file: {e}")


def create_llm_instance(
    config_path: str = "configs/config.yaml", temperature: Optional[float] = None
) -> Union[GigaChat, ChatOpenAI]:
    """
    Creates LLM instance based on config.
    If temperature is None, it loads from config.
    """
    cfg = load_config(config_path)

    # Priority: Function Argument > Config File > Default (0.1)
    if temperature is None:
        temperature = cfg.get("temperature", 0.1)

    provider = cfg.get("provider", "gigachat").lower()

    if provider == "gigachat":
        return GigaChat(
            base_url=cfg.get("BASE_URL"),
            cert_file=cfg.get("CERT_FILE"),
            key_file=cfg.get("KEY_FILE"),
            model=cfg.get("MODEL_NAME"),
            verify_ssl_certs=cfg.get("VERIFY_SSL_CERTS", False),
            profanity_check=cfg.get("PROFANITY_CHECK", False),
            timeout=int(cfg.get("TIMEOUT", 30)),
            credentials=cfg.get("AUTH_DATA"),
            scope=cfg.get("SCOPE", "GIGACHAT_API_PERS"),
            temperature=temperature,
        )

    elif provider == "openai":
        model_name = cfg.get("OPENAI_MODEL_NAME") or cfg.get("MODEL_NAME")
        return ChatOpenAI(
            model=model_name,
            api_key=cfg.get("OPENAI_API_KEY"),
            base_url=cfg.get("OPENAI_API_BASE"),
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown provider: '{provider}'")
