import uvicorn
import time
import base64
import os
import json
import requests
import numpy as np
import cv2
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, ConfigDict
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.sse import SseServerTransport
from collections import Counter
from starlette.routing import Route

# 공식 SDK 모델 우회용 (Pydantic에서 지워지더라도 통과시키기 위함)
Tool.model_config = ConfigDict(extra="allow")

class PlayMCPTool(Tool):
    annotations: dict = Field(default_factory=dict)

server = Server("mcp-vision-guide")
MODEL_PATH = "efficientdet_lite0.tflite"

# ==========================================
# 1. AI 모델 초기화 (초고속 부팅을 위해 임시 주석)
# ==========================================
detector = None
print("✅ [디버그 모드] AI 모델 로드 생략. 덧셈 툴만 활성화됩니다.")

# ==========================================
# 2. MCP 툴 리스트 정의
# ==========================================
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    # 신규 추가: 초고속 응답 테스트 툴 (숫자 곱셈)
    tool_multiply = PlayMCPTool(
        name="fast_multiply_test",
        description="A lightweight tool to multiply two numbers and test sub-3-second response times. Provided by VisionHelper(비전헬퍼).",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    return [tool_multiply]

# ==========================================
# 5. MCP 툴 라우터 (핵심 비전 로직 생략)
# ==========================================
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    start_time = time.perf_counter()
    
    if name == "fast_multiply_test":
        try:
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            result = a * b
            result_text = f"✅ 빠른 테스트 결과: {a} * {b} = {result}"
        except (TypeError, ValueError):
            result_text = "⚠️ 오류: 유효한 숫자가 입력되지 않았습니다."
    else:
        raise ValueError(f"Unknown tool: {name}")

    latency_ms = (time.perf_counter() - start_time) * 1000
    if latency_ms > 3000:
        print("🚨 [WARNING] p99 3000ms 초과 발생!")
    
    return [TextContent(type="text", text=result_text)]

# ==========================================
# 6. FastAPI 및 라우팅 설정 (최종 가로채기 방어 적용)
# ==========================================
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

# ✨ [핵심 방어 로직] Pydantic이 지워버린 annotations를 통신 직전에 강제 복구합니다.
async def handle_sse(request: Request):
    original_send = request._send
    
    async def intercepted_send(message: dict):
        if message.get("type") == "http.response.body" and b"tools" in message.get("body", b""):
            try:
                body_str = message["body"].decode("utf-8")
                # SDK가 생성한 SSE 스트림 문자열을 가로챕니다.
                if body_str.startswith("event: message\ndata: "):
                    parts = body_str.split("data: ", 1)
                    prefix = parts[0] + "data: "
                    json_str, tail = parts[1].rsplit("\n\n", 1)
                    
                    data = json.loads(json_str)
                    
                    # 툴 목록에 카카오 필수 annotations 강제 주입
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
                    new_body_str = f"{prefix}{new_json_str}\n\n{tail}"
                    message["body"] = new_body_str.encode("utf-8")
            except Exception as e:
                print("🚨 Intercept Error:", e)
                
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