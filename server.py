import uvicorn
import time
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, ConfigDict
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.sse import SseServerTransport
from starlette.routing import Route

Tool.model_config = ConfigDict(extra="allow")

class PlayMCPTool(Tool):
    annotations: dict = Field(default_factory=dict)

server = Server("mcp-vision-guide")

# 1. 덧셈 툴만 단독 활성화 (가장 안전하고 빠른 테스트용)
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
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

# 프론트엔드 통신 허용 (CORS)
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

# ✨ [핵심 방어] 프록시 관통 및 카카오 필수 규격 강제 주입
async def handle_sse(request: Request):
    original_send = request._send
    
    async def intercepted_send(message: dict):
        # 1. 클라우드 프록시 버퍼링 강제 차단 (연결 실패 해결)
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((b"x-accel-buffering", b"no"))
            message["headers"] = headers

        # 2. Pydantic이 지워버린 annotations 강제 복구 (툴 미노출 해결)
        if message.get("type") == "http.response.body" and b"tools" in message.get("body", b""):
            try:
                body_str = message["body"].decode("utf-8")
                if body_str.startswith("event: message\ndata: "):
                    parts = body_str.split("data: ", 1)
                    prefix = parts[0] + "data: "
                    json_str, tail = parts[1].rsplit("\n\n", 1)
                    
                    data = json.loads(json_str)
                    if "result" in data and "tools" in data["result"]:
                        for tool in data["result"]["tools"]:
                            tool["annotations"] = {
                                "title": tool.get("name", "Tool"),
                                "readOnlyHint": True,
                                "destructiveHint": False,
                                "openWorldHint": False,
                                "idempotentHint": True
                            }
                    
                    new_json_str = json.dumps(data)
                    message["body"] = f"{prefix}{new_json_str}\n\n{tail}".encode("utf-8")
            except Exception:
                pass
                
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