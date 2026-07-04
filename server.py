import uvicorn
import time
import base64
import os
import requests
import numpy as np
import cv2
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # ✨ CORS 방어용 추가
from pydantic import Field, ConfigDict
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.sse import SseServerTransport
from collections import Counter
from starlette.routing import Route

# [핵심 방어] 공식 MCP SDK가 카카오 전용 필드(annotations)를 삭제하지 못하도록 강제 허용
Tool.model_config = ConfigDict(extra="allow")

# PlayMCP 규정에 맞춘 필수 필드 (annotations 포함)
class PlayMCPTool(Tool):
    annotations: dict = Field(default_factory=dict)

server = Server("mcp-vision-guide")
MODEL_PATH = "efficientdet_lite0.tflite"

# ==========================================
# 1. AI 모델 초기화 (디버깅을 위해 임시 주석 처리 - 초고속 부팅)
# ==========================================
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    print("✅ MediaPipe 모델 로드 완료! (초고속 추론 준비 완료)")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    detector = None
"""
detector = None
print("✅ [디버그 모드] AI 모델 로드 생략. 덧셈 툴만 활성화됩니다.")

# ==========================================
# 2. MCP 툴 리스트 정의
# ==========================================
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    # ✨ 디버깅을 위해 Base64 비전 툴 임시 주석 처리
    tool_base64 = PlayMCPTool(
        name="analyze_vision_base64_optimized",
        description="Analyzes an optimized (compressed under 640px) Base64 image with ultra-low latency for visually impaired real-time assistance. Provided by VisionHelper(비전헬퍼).",
        inputSchema={
            "type": "object",
            "properties": {"image_base64": {"type": "string", "description": "Edge-compressed Base64 string"}},
            "required": ["image_base64"]
        },
        annotations={
            "title": "Base64 Image Vision Analysis",
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": True
        }
    )
    """
    
    # ✨ 신규 추가: 초고속 응답 테스트 툴 (숫자 곱셈)만 단독 활성화
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
        },
        annotations={
            "title": "Fast Multiply Test",
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": True
        }
    )
    
    # 오직 곱셈 툴 하나만 반환합니다.
    return [tool_multiply]

# ==========================================
# 3. 핵심 AI 분석 로직 (호출되지 않으므로 그대로 둠)
# ==========================================
def process_core_logic(img_cv2: np.ndarray) -> str:
    return "⚠️ 현재 디버그 모드(비전 기능 비활성화)입니다."

# ==========================================
# 4. 데이터 소스 처리 로직 (호출되지 않으므로 그대로 둠)
# ==========================================
def process_base64_sync(image_base64: str) -> str:
    return "⚠️ 현재 디버그 모드(비전 기능 비활성화)입니다."

# ==========================================
# 5. MCP 툴 라우터
# ==========================================
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    start_time = time.perf_counter()

    """
    # ✨ 비전 툴 호출 차단
    if name == "analyze_vision_base64_optimized":
        base64_str = arguments.get("image_base64", "")
        if not base64_str: return [TextContent(type="text", text="⚠️ 오류: Base64가 입력되지 않았습니다.")]
        result_text = await asyncio.to_thread(process_base64_sync, base64_str)
    """
    
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
# 6. FastAPI 및 라우팅 설정 (완벽 방어 적용)
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

# ✨ [핵심 방어] FastAPI의 'NoneType' 에러를 막기 위한 빈(Empty) ASGI 함수
async def empty_asgi_app(scope, receive, send):
    pass

async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    # ✨ 에러 방지용 더미 앱 반환
    return empty_asgi_app

async def handle_post(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)
    # ✨ 에러 방지용 더미 앱 반환
    return empty_asgi_app

# 프록시 경로 잘림 대비 (/mcp 및 / 모두 허용)
app.routes.append(Route("/mcp", endpoint=handle_sse, methods=["GET"]))
app.routes.append(Route("/mcp", endpoint=handle_post, methods=["POST"]))
app.routes.append(Route("/", endpoint=handle_sse, methods=["GET"]))
app.routes.append(Route("/", endpoint=handle_post, methods=["POST"]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    # 클라우드 환경 접속 차단 방지 (proxy_headers)
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")