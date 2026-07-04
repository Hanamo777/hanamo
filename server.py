import uvicorn
import time
import base64
import os
import requests
import numpy as np
import cv2
import mediapipe as mp
import asyncio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
# 1. AI 모델 초기화 (메모리 로드)
# ==========================================
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    print("✅ MediaPipe 모델 로드 완료! (초고속 추론 준비 완료)")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    detector = None

# ==========================================
# 2. MCP 툴 리스트 정의
# ==========================================
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    # PlayMCP 규정에 따라 영문 작성 권장 및 서비스명(영/한) 병기, annotations 속성 5가지 필수 포함
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
    
    return [tool_base64]

# ==========================================
# 3. 핵심 AI 분석 로직 (순수 함수)
# ==========================================
def process_core_logic(img_cv2: np.ndarray) -> str:
    if detector is None:
        return "⚠️ **오류**: AI 시각 모델이 로드되지 않았습니다."

    max_dim = 640
    h, w = img_cv2.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_cv2 = cv2.resize(img_cv2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    img_height, img_width, _ = img_cv2.shape
    total_image_area = img_width * img_height

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return "🔍 전방에 뚜렷하게 인식되는 물체가 없습니다. 필요시 다른 방향을 촬영해 주세요."

    detected_names = []
    collision_warnings = []
    directional_guidance = []

    for detection in detection_result.detections:
        obj_name = detection.categories[0].category_name
        detected_names.append(obj_name)
        
        bbox = detection.bounding_box
        box_area = bbox.width * bbox.height
        center_x = bbox.origin_x + (bbox.width / 2)

        pos_ratio = center_x / img_width
        if pos_ratio < 0.2: direction = "10시 방향 (좌측 끝)"
        elif pos_ratio < 0.4: direction = "11시 방향 (좌측)"
        elif pos_ratio < 0.6: direction = "12시 방향 (정면)"
        elif pos_ratio < 0.8: direction = "1시 방향 (우측)"
        else: direction = "2시 방향 (우측 끝)"
        
        directional_guidance.append(f"- **{obj_name}**: {direction}")

        area_ratio = box_area / total_image_area
        if area_ratio >= 0.5:
            if pos_ratio < 0.4: safe_action = "오른쪽(1~2시 방향)으로 피하시거나"
            elif pos_ratio > 0.6: safe_action = "왼쪽(10~11시 방향)으로 피하시거나"
            else: safe_action = "뒤로 물러서서 비어있는 길을 확인해"
            collision_warnings.append(f"- ⚠️ **[긴급] {direction}**에 **{obj_name}**이(가) 있습니다! {safe_action} 주세요.")

    counts = Counter(detected_names)
    counts_str = ", ".join([f"{item} {count}개" for item, count in counts.items()])

    result_text = f"### 👁️ VisionHelper(비전헬퍼) 시각보조 분석 보고서\n\n"
    if collision_warnings:
        result_text += f"#### 🚨 긴급 충돌 경고\n" + "\n".join(collision_warnings) + "\n\n"
    else:
        result_text += "#### ✅ 안전 상태\n현재 50% 이상 면적을 차지하는 초근접 위험 장애물은 없습니다.\n\n"

    result_text += f"#### 📊 탐지 결과 요약\n- **총합**: {counts_str}\n\n"
    result_text += "#### 🧭 세부 방향 위치\n" + "\n".join(directional_guidance)

    return result_text

# ==========================================
# 4. 데이터 소스 처리 로직
# ==========================================
def process_base64_sync(image_base64: str) -> str:
    try:
        if "," in image_base64: image_base64 = image_base64.split(",")[1]
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv2 is None: return "⚠️ 오류: 손상된 Base64 데이터입니다."
        return process_core_logic(img_cv2)
    except Exception as e:
        return f"⚠️ Base64 이미지 분석 실패: {str(e)}"

# ==========================================
# 5. MCP 툴 라우터
# ==========================================
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    start_time = time.perf_counter()

    if name == "analyze_vision_base64_optimized":
        base64_str = arguments.get("image_base64", "")
        if not base64_str: return [TextContent(type="text", text="⚠️ 오류: Base64가 입력되지 않았습니다.")]
        result_text = await asyncio.to_thread(process_base64_sync, base64_str)
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

# ✨ PlayMCP 대시보드에서 툴 목록을 읽어갈 수 있도록 CORS 완벽 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sse = SseServerTransport("/mcp")

# 상태 체크용 기본 루트
@app.get("/health")
def health_check():
    return {"status": "Active"}

# 🚨 기존에 있던 @app.get("/{path:path}") 와 @app.post("/{path:path}") 부분은 완전히 지워주세요.

# ✨ FastAPI가 응답을 생성하지 못하도록 순수 비동기 함수로만 정의합니다.
async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

async def handle_post(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

# ✨ Starlette Route를 사용하여 "/mcp" 경로에 직접 연결합니다.
app.routes.append(Route("/mcp", endpoint=handle_sse, methods=["GET"]))
app.routes.append(Route("/mcp", endpoint=handle_post, methods=["POST"]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)