from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uuid
import base64
import numpy as np
import cv2
import mediapipe as mp
import requests
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

app = FastAPI(title="Vision Guide Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SERVICE_NAME = "HelpTheBlind(헬프더블라인드)"
SUPPORTED_VERSIONS = ["2025-11-25", "2025-03-26"]

active_sessions = {}

# ==========================================
# 1. AI 모델 초기화
# ==========================================
MODEL_PATH = "efficientdet_lite0.tflite"
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    print("✅ MediaPipe 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    detector = None

# ==========================================
# 2. 이미지 분석 코어 로직
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

    result_text = f"### 👁️ {SERVICE_NAME} 시각보조 분석 보고서\n\n"
    if collision_warnings:
        result_text += f"#### 🚨 긴급 충돌 경고\n" + "\n".join(collision_warnings) + "\n\n"
    else:
        result_text += "#### ✅ 안전 상태\n현재 50% 이상 면적을 차지하는 초근접 위험 장애물은 없습니다.\n\n"

    result_text += f"#### 📊 탐지 결과 요약\n- **총합**: {counts_str}\n\n"
    result_text += "#### 🧭 세부 방향 위치\n" + "\n".join(directional_guidance)

    return result_text

# --- 추후 사용을 위한 Base64 로직 주석 처리 ---
# def process_base64_sync(image_base64: str) -> str:
#     try:
#         if "," in image_base64: image_base64 = image_base64.split(",")[1]
#         img_data = base64.b64decode(image_base64)
#         nparr = np.frombuffer(img_data, np.uint8)
#         img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img_cv2 is None: return "⚠️ 오류: 손상된 Base64 데이터입니다."
#         return process_core_logic(img_cv2)
#     except Exception as e:
#         return f"⚠️ Base64 이미지 분석 실패: {str(e)}"

# ==========================================
# 2-1. URL 기반 이미지 다운로드 및 처리 로직
# ==========================================
def convert_gdrive_url(url: str) -> str:
    """구글 드라이브 View 링크를 직접 다운로드 링크로 자동 변환"""
    match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def process_url_sync(image_url: str) -> str:
    try:
        # 구글 드라이브 링크 호환 처리
        direct_url = convert_gdrive_url(image_url)
        
        # 일부 서버에서 봇을 차단하는 것을 막기 위해 User-Agent 헤더 추가
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(direct_url, headers=headers, timeout=2.5)
        response.raise_for_status()
        
        # 바이트 데이터를 OpenCV 이미지로 디코딩
        nparr = np.frombuffer(response.content, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            return "⚠️ 오류: URL에서 유효한 이미지를 읽을 수 없습니다. (접근 권한이 없거나 지원하지 않는 파일 형식일 수 있습니다.)"
            
        return process_core_logic(img_cv2)
    except Exception as e:
        return f"⚠️ 이미지 URL 다운로드 및 분석 실패: {str(e)}"

# ==========================================
# 3. MCP 핸들러
# ==========================================
def handle_initialize(req_id: str | int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mcp-vision-guide", "version": "1.0.0"}
        }
    }

def handle_tools_list(req_id: str | int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "analyze_vision_url",
                    "description": f"Analyzes an image from a given public URL (including Google Drive links) for visually impaired real-time assistance. Provided by {SERVICE_NAME}.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string", "description": "Public URL of the image"}},
                        "required": ["image_url"]
                    },
                    "annotations": {
                        "title": "Vision Analysis by URL",
                        "readOnlyHint": True,
                        "destructiveHint": False,
                        "openWorldHint": True, # 외부 인터넷(URL)과 통신하므로 True로 변경
                        "idempotentHint": True
                    }
                }
                # --- 기존 툴들 주석 처리 ---
                # {
                #     "name": "analyze_vision_base64_optimized",
                #     "description": f"Analyzes an optimized Base64 image... Provided by {SERVICE_NAME}.",
                #     "inputSchema": { ... },
                #     "annotations": { ... }
                # },
                # {
                #     "name": "fast_multiply_test",
                #     ...
                # }
            ]
        }
    }

async def handle_tools_call(req_id: str | int, params: dict) -> dict:
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        if tool_name == "analyze_vision_url":
            image_url = args.get("image_url", "")
            if not image_url:
                text_content = "⚠️ 오류: 이미지 URL이 입력되지 않았습니다."
            else:
                text_content = await asyncio.to_thread(process_url_sync, image_url)
                
        # --- 기존 툴 실행 로직 주석 처리 ---
        # elif tool_name == "analyze_vision_base64_optimized":
        #     ...
        # elif tool_name == "fast_multiply_test":
        #     ...
            
        else:
            text_content = f"Error: 알 수 없는 툴 이름입니다. ({tool_name})"
    except Exception as e:
        text_content = f"Error: 처리 중 오류가 발생했습니다. ({str(e)})"

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text_content}]
        }
    }

# ==========================================
# 4. 엔드포인트 라우팅 (SSE 연결 로직 유지)
# ==========================================
@app.get("/mcp")
async def mcp_get_endpoint(request: Request):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    active_sessions[session_id] = queue

    async def event_generator():
        yield f"event: endpoint\ndata: /mcp?sessionId={session_id}\n\n"
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            if session_id in active_sessions:
                del active_sessions[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp")
async def mcp_post_endpoint(request: Request, sessionId: str = None):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    if method == "initialize":
        response_data = handle_initialize(req_id)
    elif method == "tools/list":
        response_data = handle_tools_list(req_id)
    elif method == "tools/call":
        response_data = await handle_tools_call(req_id, params)
    elif method == "notifications/initialized":
        return Response(status_code=202)
    else:
        response_data = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    if sessionId and sessionId in active_sessions:
        await active_sessions[sessionId].put(response_data)
        return Response(status_code=202)
    else:
        return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)