# Venice Bridge 🚀

Ponte de automação para usar sua assinatura do **Venice AI Pro** (chat ilimitado e modelos privados/sem censura) localmente no seu ambiente de desenvolvimento através de:
1. **MCP Server** (Model Context Protocol) para Antigravity, Claude Code, Cursor, etc.
2. **API OpenAI-Compatible** (`http://127.0.0.1:8000/v1/chat/completions`).

A ponte utiliza o [`agent-browser`](https://github.com/agent-browser) para manter uma sessão segura com perfil persistente do Chrome, comunicando-se diretamente com os endpoints de inferência do Venice AI sem custos adicionais de tokens de API.

---

## 🛠️ Pré-requisitos

- Python `>= 3.12` com [`uv`](https://github.com/astral-sh/uv)
- `agent-browser` instalado em `~/.local/bin/agent-browser`
- Perfil do navegador configurado em `~/.venice-browser-profile` com login efetuado no Venice.

---

## 📦 Instalação

No diretório do projeto:

```bash
cd ~/Developer/obrunogonzaga/venice-bridge
uv sync
uv pip install -e .
```

---

## 🔌 1. Configurando como MCP Server

O servidor MCP expõe 4 ferramentas principais:
* `venice_chat(prompt, model?, system_prompt?, web_enabled?)`: Conversa com a IA no Venice.
* `venice_list_models()`: Lista todos os modelos disponíveis e sinaliza quais são ilimitados no plano Pro.
* `venice_status()`: Exibe seu perfil autenticado e contagem de rate limits restantes (ex: 9.970+ chats restantes).
* `venice_generate_image(prompt, model?, width?, height?)`: Gera imagens com modelos Pro gratuitos (`chroma`, `z-image-turbo`, `wai-illu`) e salva em disco.

### Configuração no Antigravity / Cursor / Claude:

Adicione ao seu arquivo de configuração MCP (ex: `~/.gemini/antigravity-cli/mcp_config.json` ou similar):

```json
{
  "mcpServers": {
    "venice-bridge": {
      "command": "/opt/homebrew/bin/uv",
      "args": [
        "--directory",
        "/Users/bruno/Developer/obrunogonzaga/venice-bridge",
        "run",
        "venice-mcp"
      ]
    }
  }
}
```

---

## 🌐 2. Executando como API OpenAI Local

Para usar com ferramentas que consom endpoints OpenAI (Continue, Cline, OpenWebUI, scripts python, curl):

```bash
cd ~/Developer/obrunogonzaga/venice-bridge
uv run venice-api
```

O servidor iniciará em `http://127.0.0.1:8000`:
- **Base URL:** `http://127.0.0.1:8000/v1`
- **Listar Modelos:** `GET /v1/models`
- **Chat:** `POST /v1/chat/completions`

### Exemplo via cURL:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "venice-uncensored-1-2",
    "messages": [
      {"role": "user", "content": "Olá Venice!"}
    ]
  }'
```

---

## 🧠 Modelos em Destaque

| Modelo | ID | Uso no Plano Pro |
| :--- | :--- | :--- |
| **Venice Uncensored 1.2** | `venice-uncensored-1-2` | **Ilimitado (Gratuito)** |
| **DeepSeek V4.1 Flash Uncensored** | `deepseek-v4-1-flash-uncensored` | **Ilimitado (Gratuito)** |
| **Auto** | `default` | **Ilimitado (Gratuito)** |
| **Chroma (Imagem)** | `chroma` | **4.000 imgs/dia (Gratuito)** |
| **Z-Image Turbo** | `z-image-turbo` | **4.000 imgs/dia (Gratuito)** |

---

## 📁 Estrutura do Código

```
venice-bridge/
├── pyproject.toml
├── src/
│   └── venice_bridge/
│       ├── __init__.py
│       ├── client.py       # Driver do Venice conectado via agent-browser
│       ├── server_mcp.py   # Servidor FastMCP/MCPServer
│       └── server_api.py   # Servidor FastAPI compatível com OpenAI
└── outputs/                # Imagens geradas localmente
```
