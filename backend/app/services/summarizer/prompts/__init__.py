"""
Prompt templates for LLM summarization.

IMPORTANT: Prompt templates are versioned and FROZEN once deployed.
Never edit a frozen template - create a new version instead.

This ensures reproducibility of summaries across time.
"""

from .v1 import (
    PROMPT_VERSION as V1_PROMPT_VERSION,
    SYSTEM_PROMPT as V1_SYSTEM_PROMPT,
    build_user_prompt as v1_build_user_prompt,
)

# Current default version
CURRENT_PROMPT_VERSION = V1_PROMPT_VERSION
CURRENT_SYSTEM_PROMPT = V1_SYSTEM_PROMPT
build_current_user_prompt = v1_build_user_prompt

__all__ = [
    "CURRENT_PROMPT_VERSION",
    "CURRENT_SYSTEM_PROMPT",
    "build_current_user_prompt",
    "V1_PROMPT_VERSION",
    "V1_SYSTEM_PROMPT",
    "v1_build_user_prompt",
]
