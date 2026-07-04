import uvicorn
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, ConfigDict
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.sse import SseServerTransport
from starlette.routing import Route

# Pydantic이 추가 필드(annotations)를 직렬화할 수 있도록 허용
Tool.model_config = ConfigDict(extra="allow")

class PlayMCPTool(Tool):
    annotations: dict = Field(default_factory=dict)

server = Server("mcp-vision-guide")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    # ✨ 핵심: 불안정한 통신 조작 대신, 여기서 툴을 만들 때 바로 annotations를 주입합니다.
    tool_multiply = PlayMCPTool(
        name="fast_multiply_test",
        description="A lightweight tool to multiply two numbers. Provided by VisionHelper(비전헬퍼).",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        annotations={
            "title": "fast_multiply_test",
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": True
        }
    )
    return [tool_multiply]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "fast_multiply_test":
        try:
            result = float(arguments.get("a", 0)) * float(arguments.get("b", 0))
            return [TextContent(type="text", text=f"✅ 결과: {result}")]
        except:
            return [TextContent(type="text", text="⚠️ 오류")]
    raise ValueError(f"Unknown tool: {name}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sse = SseServerTransport("/mcp")

@app.get("/health")
def health_check():
    return {"status": "Active"}

async def empty_asgi_app(scope, receive, send):
    pass

# ✨ [최종 방어] 텍스트 파싱/버퍼링 같은 불안정한 조작은 싹 다 지우고, 프록시 버퍼링 차단만 남겼습니다.
async def handle_sse(request: Request):
    original_send = request._send

    async def intercepted_send(message: dict):
        # 카카오 클라우드 프록시(Envoy/Nginx)의 연결 끊김을 막기 위한 필수 헤더
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((b"x-accel-buffering", b"no"))
            headers.append((b"cache-control", b"no-cache"))
            message["headers"] = headers
            
        # 본문(Body)은 손대지 않고, Pydantic이 만든 순정 상태 그대로 즉시 전송
        await original_send(message)

    async with sse.connect_sse(request.scope, request.receive, intercepted_send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return empty_asgi_app

async def handle_post(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)
    return empty_asgi_app

app.routes.append(Route("/mcp", endpoint=handle_sse, methods=["GET"]))
app.routes.append(Route("/mcp", endpoint=handle_post, methods=["POST"]))
app.routes.append(Route("/", endpoint=handle_sse, methods=["GET"]))
app.routes.append(Route("/", endpoint=handle_post, methods=["POST"]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")