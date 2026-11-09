"""
Venice Bridge MCP Server
Exposes Venice AI capabilities to AI agents via Model Context Protocol.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Optional

from mcp.server.mcpserver import MCPServer
from venice_bridge.client import VeniceClient, VeniceError

mcp = MCPServer("venice-bridge")
client = VeniceClient()


@mcp.tool()
def venice_chat(
    prompt: str,
    model: str = "deepseek-v4-1-flash-uncensored",
    system_prompt: str = "",
    web_enabled: bool = False,
) -> str:
    """
    Sends a prompt to Venice AI using the active Pro subscription.
    
    :param prompt: The user prompt or instruction.
    :param model: Model ID to use. Default is 'deepseek-v4-1-flash-uncensored'.
                  Another popular free model is 'venice-uncensored-1-2'.
    :param system_prompt: Optional custom system instructions for the model.
    :param web_enabled: Whether to enable web search grounding.
    :return: The generated response text.
    """
    try:
        res = client.chat(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            web_enabled=web_enabled,
        )
        return res["text"]
    except VeniceError as e:
        return f"Error communicating with Venice AI: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


@mcp.tool()
def venice_list_models() -> str:
    """
    Lists available models in Venice AI (chat and image models)
    along with their Pro credit usage (free vs uses credits).
    """
    try:
        data = client.list_models()
        chat_models = data.get("agentModelOptions", [])
        image_gen_models = data.get("imageModelOptions", {}).get("generation", [])

        lines = ["# Modelos Venice AI Disponíveis\n"]
        lines.append("## Modelos de Chat:")
        for m in chat_models:
            free_tag = " [ILIMITADO NO PRO]" if not m.get("usesCredits") else " [Consome Créditos]"
            lines.append(f"- **{m.get('name')}** (`{m.get('id')}`){free_tag}")

        lines.append("\n## Modelos de Geração de Imagem:")
        for m in image_gen_models:
            free_tag = " [ILIMITADO NO PRO]" if not m.get("usesCredits") else " [Consome Créditos]"
            lines.append(f"- **{m.get('name')}** (`{m.get('id')}`){free_tag}")

        return "\n".join(lines)
    except Exception as e:
        return f"Erro ao listar modelos: {e}"


@mcp.tool()
def venice_status() -> str:
    """
    Returns Venice account status, active session info, and remaining quotas/rate limits.
    """
    try:
        status = client.get_status()
        limits = client.get_rate_limits()

        if not status.get("loggedIn"):
            return "Sessão Venice não autenticada. Abra o navegador com `agent-browser --session venice` para efetuar login."

        chat_limits = limits.get("chat", {})
        image_limits = limits.get("image", {})

        return (
            f"### Status da Conta Venice\n"
            f"- **Usuário:** {status.get('fullName')} ({status.get('email')})\n"
            f"- **User ID:** `{status.get('userId')}`\n"
            f"- **Chats restantes hoje:** {chat_limits.get('remaining', 'N/A')}\n"
            f"- **Imagens restantes hoje:** {image_limits.get('remaining', 'N/A')}\n"
            f"- **Status da Sessão:** Ativa e conectada via agent-browser\n"
        )
    except Exception as e:
        return f"Erro ao obter status do Venice: {e}"


@mcp.tool()
def venice_generate_image(
    prompt: str,
    model: str = "chroma",
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Generates an image via Venice AI using free Pro models ('chroma', 'z-image-turbo', 'wai-illu')
    and saves it to disk. Returns the saved local file path and a markdown image preview.
    
    :param prompt: Image prompt description.
    :param model: Image model ID (default 'chroma').
    :param width: Image width (e.g. 512, 1024).
    :param height: Image height (e.g. 512, 1024).
    :return: File path and markdown preview.
    """
    try:
        res = client.generate_image(prompt=prompt, model=model, width=width, height=height)
        img_bytes = base64.b64decode(res["base64"])

        output_dir = os.path.expanduser("~/Developer/obrunogonzaga/venice-bridge/outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"venice_{int(time.time())}_{model}.webp"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        return (
            f"Imagem gerada com sucesso!\n"
            f"- Modelo: {model}\n"
            f"- Dimensões: {width}x{height}\n"
            f"- Caminho salvo: {filepath}\n"
            f"\n![Imagem Gerada]({filepath})"
        )
    except Exception as e:
        return f"Erro ao gerar imagem: {e}"


def main():
    """Main entrypoint for MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
