"""Memory management utilities for Honcho-backed conversation memory.

This module provides helper functions and classes for storing, retrieving,
and summarizing conversation memory using Honcho's metamessage and collection APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from honcho import AsyncHoncho
from honcho.types import Message, Metamessage

logger = logging.getLogger(__name__)

# Metamessage type labels used to tag memory entries
MEMORY_TYPE_SUMMARY = "summary"
MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_PREFERENCE = "preference"


@dataclass
class MemoryEntry:
    """A single piece of extracted memory associated with a user."""

    content: str
    memory_type: str = MEMORY_TYPE_FACT
    metadata: dict = field(default_factory=dict)


async def store_memory(
    client: AsyncHoncho,
    app_id: str,
    user_id: str,
    session_id: str,
    message: Message,
    memory: MemoryEntry,
) -> Metamessage:
    """Attach a memory entry to an existing message as a metamessage.

    Args:
        client: Authenticated AsyncHoncho client.
        app_id: Honcho application identifier.
        user_id: Honcho user identifier.
        session_id: Session the message belongs to.
        message: The message this memory was derived from.
        memory: The memory entry to store.

    Returns:
        The created Metamessage object.
    """
    metamessage = await client.apps.users.sessions.messages.metamessages.create(
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        message_id=message.id,
        metamessage_type=memory.memory_type,
        content=memory.content,
        metadata=memory.metadata,
    )
    logger.debug(
        "Stored memory (type=%s) for user=%s session=%s",
        memory.memory_type,
        user_id,
        session_id,
    )
    return metamessage


async def fetch_memories(
    client: AsyncHoncho,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    memory_type: Optional[str] = None,
) -> list[Metamessage]:
    """Retrieve all metamessages (memories) attached to a specific message.

    Args:
        client: Authenticated AsyncHoncho client.
        app_id: Honcho application identifier.
        user_id: Honcho user identifier.
        session_id: Session the message belongs to.
        message_id: The message whose metamessages should be fetched.
        memory_type: Optional filter; if provided only metamessages of this
            type are returned.

    Returns:
        List of Metamessage objects, possibly empty.
    """
    params: dict = {}
    if memory_type:
        params["metamessage_type"] = memory_type

    page = await client.apps.users.sessions.messages.metamessages.list(
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        **params,
    )
    results: list[Metamessage] = []
    async for item in page:
        results.append(item)

    logger.debug(
        "Fetched %d memories for message=%s (type=%s)",
        len(results),
        message_id,
        memory_type or "*",
    )
    return results


def format_memories_for_prompt(memories: list[Metamessage]) -> str:
    """Render a list of metamessages into a concise prompt-ready string.

    Memories are grouped by type and presented as a bulleted list so they
    can be injected directly into a system prompt.

    Args:
        memories: Metamessage objects to format.

    Returns:
        A multi-line string, or an empty string if no memories are provided.
    """
    if not memories:
        return ""

    grouped: dict[str, list[str]] = {}
    for m in memories:
        grouped.setdefault(m.metamessage_type, []).append(m.content)

    lines: list[str] = []
    for mtype, contents in grouped.items():
        lines.append(f"[{mtype.upper()}]")
        for content in contents:
            lines.append(f"  - {content}")

    return "\n".join(lines)
