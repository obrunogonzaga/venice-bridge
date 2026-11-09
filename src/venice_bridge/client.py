"""
Venice Bridge Client
Communicates with Venice AI via an agent-browser session.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("venice_bridge.client")

DEFAULT_PROFILE = os.path.expanduser("~/.venice-browser-profile")
DEFAULT_SESSION = "venice"
VENICE_CHAT_URL = "https://venice.ai/chat/classic"
OUTERFACE_BASE_URL = "https://outerface.venice.ai"


class VeniceError(Exception):
    """Base exception for Venice Bridge errors."""
    pass


class VeniceClient:
    """Client for automating Venice AI requests via agent-browser."""

    def __init__(
        self,
        session_name: str = DEFAULT_SESSION,
        profile_path: str = DEFAULT_PROFILE,
        agent_browser_bin: str = "agent-browser",
    ):
        self.session_name = session_name
        self.profile_path = profile_path
        self.bin = agent_browser_bin

    def _run_cmd(self, args: List[str], timeout: float = 60.0) -> str:
        """Executes an agent-browser command."""
        cmd = [self.bin, "--session", self.session_name] + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if res.returncode != 0:
                err_msg = res.stderr.strip() or res.stdout.strip()
                raise VeniceError(f"agent-browser command failed: {err_msg}")
            return res.stdout.strip()
        except subprocess.TimeoutExpired:
            raise VeniceError(f"agent-browser command timed out after {timeout}s")
        except FileNotFoundError:
            raise VeniceError(f"Binary '{self.bin}' not found in PATH")

    def eval_js(self, js_code: str, timeout: float = 60.0) -> Any:
        """Evaluates JavaScript code inside the Venice browser tab and returns parsed JSON."""
        # Ensure the session is running before eval
        self.ensure_session()
        raw_output = self._run_cmd(["eval", js_code], timeout=timeout)
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output

    def ensure_session(self) -> bool:
        """Verifies if the Venice session is alive. If not, launches it in headless mode with profile."""
        try:
            url = self._run_cmd(["get", "url"], timeout=5.0)
            if "venice.ai" in url:
                return True
        except Exception:
            pass

        # Session inactive, start or navigate
        logger.info(f"Opening Venice browser session '{self.session_name}'...")
        self._run_cmd([
            "--profile", self.profile_path,
            "open", VENICE_CHAT_URL
        ], timeout=20.0)
        time.sleep(2.0)
        return True

    def get_status(self) -> Dict[str, Any]:
        """Gets user authentication state and account details."""
        js = """
        (async () => {
            if (typeof window.Clerk === 'undefined' || !window.Clerk.user) {
                return { loggedIn: false };
            }
            return {
                loggedIn: true,
                userId: window.Clerk.user.id,
                email: window.Clerk.user.primaryEmailAddress?.emailAddress,
                fullName: window.Clerk.user.fullName,
                sessionTokenExp: window.Clerk.session?.expireAt
            };
        })()
        """
        data = self.eval_js(js)
        if isinstance(data, dict):
            return data
        return {"loggedIn": False, "raw": data}

    def get_rate_limits(self) -> Dict[str, Any]:
        """Fetches current Venice rate limits and quotas."""
        js = """
        (async () => {
            const token = await window.Clerk.session.getToken();
            const res = await fetch('https://outerface.venice.ai/api/inference/rate-limits', {
                headers: {
                    'Authorization': 'Bearer ' + token,
                    'x-venice-version': 'interface@20261109.004936+14a157a',
                    'x-venice-middleface-version': '0.1.1219'
                }
            });
            return { status: res.status, data: await res.json() };
        })()
        """
        res = self.eval_js(js)
        if isinstance(res, dict) and res.get("status") == 200:
            return res.get("data", {})
        raise VeniceError(f"Failed to fetch rate limits: {res}")

    def list_models(self) -> Dict[str, Any]:
        """Lists available chat and image models."""
        js = """
        (async () => {
            const token = await window.Clerk.session.getToken();
            const res = await fetch('https://outerface.venice.ai/api/inference/workflow/chat/model_options', {
                headers: {
                    'Authorization': 'Bearer ' + token,
                    'x-venice-version': 'interface@20261109.004936+14a157a',
                    'x-venice-middleface-version': '0.1.1219'
                }
            });
            if (!res.ok) return { error: await res.text(), status: res.status };
            return await res.json();
        })()
        """
        return self.eval_js(js)

    def chat(
        self,
        prompt: str,
        model: str = "venice-uncensored-1-2",
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        web_enabled: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a chat prompt to Venice AI and returns the full assistant response.
        
        :param prompt: User message text (or last message if messages is provided).
        :param model: Venice modelId (e.g. 'venice-uncensored-1-2', 'deepseek-v4-1-flash-uncensored').
        :param messages: Optional list of previous chat messages [{'role': 'user'|'assistant', 'content': '...'}].
        :param system_prompt: Optional custom system instruction.
        :param temperature: Sampling temperature (0.0 to 1.0).
        :param top_p: Top-P sampling.
        :param web_enabled: Whether to enable web search.
        :return: Dict containing 'text', 'model', 'meta'.
        """
        if not messages:
            msg_list = [{"role": "user", "content": prompt}]
        else:
            msg_list = list(messages)
            if not any(m.get("content") == prompt for m in msg_list[-1:]):
                msg_list.append({"role": "user", "content": prompt})

        # Serialize parameters for JS injection
        js_params = json.dumps({
            "modelId": model,
            "messages": msg_list,
            "systemPrompt": system_prompt,
            "temperature": temperature,
            "topP": top_p,
            "webEnabled": web_enabled,
        })

        js = f"""
        (async () => {{
            const params = {js_params};
            const token = await window.Clerk.session.getToken();
            const userId = window.Clerk.user.id;

            const payload = {{
                clientProcessingTime: 1,
                conversationType: 'text',
                enableLargeContextChat: true,
                includeVeniceSystemPrompt: true,
                isCharacter: false,
                modelId: params.modelId,
                prompt: params.messages,
                reasoning: false,
                requestId: 'req_' + Math.random().toString(36).substring(2, 9),
                simpleMode: false,
                systemPrompt: params.systemPrompt,
                temperature: params.temperature,
                topP: params.topP,
                userId: userId,
                webEnabled: params.webEnabled,
                webScrapeEnabled: false,
                xSearchEnabled: false
            }};

            const res = await fetch('https://outerface.venice.ai/api/inference/chat', {{
                method: 'POST',
                headers: {{
                    'Accept': 'text/event-stream',
                    'Authorization': 'Bearer ' + token,
                    'Content-Type': 'application/json',
                    'x-venice-version': 'interface@20261109.004936+14a157a',
                    'x-venice-middleface-version': '0.1.1219'
                }},
                body: JSON.stringify(payload)
            }});

            if (!res.ok) {{
                return {{ ok: false, status: res.status, error: await res.text() }};
            }}

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let meta = {{}};
            let buffer = '';

            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, {{ stream: true }});
                const lines = buffer.split('\\n');
                buffer = lines.pop();

                for (const line of lines) {{
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    try {{
                        const data = JSON.parse(trimmed);
                        if (data.kind === 'content' && data.content) {{
                            fullText += data.content;
                        }} else if (data.kind === 'meta') {{
                            Object.assign(meta, data);
                        }}
                    }} catch (e) {{}}
                }}
            }}

            return {{ ok: true, text: fullText, meta: meta }};
        }})()
        """

        res = self.eval_js(js, timeout=120.0)
        if isinstance(res, dict) and res.get("ok"):
            return {
                "text": res.get("text", "").strip(),
                "model": model,
                "meta": res.get("meta", {}),
            }
        error_msg = res.get("error", "Unknown error") if isinstance(res, dict) else str(res)
        raise VeniceError(f"Venice chat inference failed: {error_msg}")

    def generate_image(
        self,
        prompt: str,
        model: str = "chroma",
        width: int = 1024,
        height: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generates an image via Venice AI and returns base64 image data.
        Free models on Pro: 'chroma', 'z-image-turbo', 'wai-illu'.
        """
        js_params = json.dumps({
            "prompt": prompt,
            "modelId": model,
            "width": width,
            "height": height,
        })

        js = f"""
        (async () => {{
            const params = {js_params};
            const token = await window.Clerk.session.getToken();
            const userId = window.Clerk.user.id;

            const payload = {{
                modelId: params.modelId,
                prompt: params.prompt,
                width: params.width,
                height: params.height,
                steps: 20,
                seed: Math.floor(Math.random() * 1000000),
                requestId: 'img_' + Math.random().toString(36).substring(2, 9),
                userId: userId,
                hideWatermark: false,
                matureFilter: false,
                type: 'image'
            }};

            const res = await fetch('https://outerface.venice.ai/api/inference/image', {{
                method: 'POST',
                headers: {{
                    'Authorization': 'Bearer ' + token,
                    'Content-Type': 'application/json',
                    'x-venice-version': 'interface@20261109.004936+14a157a',
                    'x-venice-middleface-version': '0.1.1219'
                }},
                body: JSON.stringify(payload)
            }});

            if (!res.ok) return {{ ok: false, status: res.status, error: await res.text() }};
            const buffer = await res.arrayBuffer();
            let binary = '';
            const bytes = new Uint8Array(buffer);
            const len = bytes.byteLength;
            for (let i = 0; i < len; i++) {{
                binary += String.fromCharCode(bytes[i]);
            }}
            const base64 = btoa(binary);
            return {{ ok: true, contentType: res.headers.get('content-type') || 'image/webp', base64: base64 }};
        }})()
        """
        res = self.eval_js(js, timeout=90.0)
        if isinstance(res, dict) and res.get("ok"):
            return {
                "base64": res["base64"],
                "content_type": res.get("contentType", "image/webp"),
                "model": model,
            }
        error_msg = res.get("error", "Unknown error") if isinstance(res, dict) else str(res)
        raise VeniceError(f"Venice image generation failed: {error_msg}")
