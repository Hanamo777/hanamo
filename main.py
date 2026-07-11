from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uuid
import numpy as np
import cv2
import mediapipe as mp
import httpx
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

SERVICE_NAME = "VisionHelper(비전헬퍼)"
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
# 2. 이미지 분석 코어 로직 (모드별 분기 추가)
# ==========================================
def process_core_logic(img_cv2: np.ndarray, mode: str = "full") -> str:
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

    # --- 요청된 모드(mode)에 따라 반환 텍스트 변경 ---
    if mode == "danger":
        if collision_warnings:
            return f"🚨 [긴급 충돌 경고]\n" + "\n".join(collision_warnings)
        return "✅ [안전 상태] 현재 초근접 위험 장애물이 감지되지 않았습니다."
        
    elif mode == "count":
        return f"📊 [객체 카운트 결과]\n총합: {counts_str}"

    # 기본(full) 보고서
    result_text = f"### 👁️ {SERVICE_NAME} 시각보조 분석 보고서\n\n"
    if collision_warnings:
        result_text += f"#### 🚨 긴급 충돌 경고\n" + "\n".join(collision_warnings) + "\n\n"
    else:
        result_text += "#### ✅ 안전 상태\n현재 50% 이상 면적을 차지하는 초근접 위험 장애물은 없습니다.\n\n"

    result_text += f"#### 📊 탐지 결과 요약\n- **총합**: {counts_str}\n\n"
    result_text += "#### 🧭 세부 방향 위치\n" + "\n".join(directional_guidance)
    return result_text

# ==========================================
# 2-1. URL 기반 이미지 처리 로직
# ==========================================
def convert_gdrive_url(url: str) -> str:
    match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

async def process_url_async(image_url: str, mode: str = "full") -> str:
    try:
        direct_url = convert_gdrive_url(image_url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }

        # 🚀 중요: follow_redirects=True 로 구글 드라이브 리다이렉트 허용, 타임아웃 2.5초
        async with httpx.AsyncClient(timeout=2.5, follow_redirects=True) as client:
            response = await client.get(direct_url, headers=headers)
            response.raise_for_status()
            image_bytes = response.content

        def decode_and_process(data: bytes) -> str:
            nparr = np.frombuffer(data, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_cv2 is None:
                return "⚠️ 오류: 다운로드된 데이터가 유효한 이미지가 아닙니다."
            return process_core_logic(img_cv2, mode=mode)
            
        return await asyncio.to_thread(decode_and_process, image_bytes)

    except httpx.TimeoutException:
        return "⚠️ [시간 초과] 이미지 다운로드에 2.5초 이상 소요되어 분석이 취소되었습니다."
    except httpx.RequestError as e:
        return f"⚠️ 네트워크 통신 오류로 이미지를 가져올 수 없습니다. ({str(e)})"
    except Exception as e:
        return f"⚠️ 이미지 분석 실패: {str(e)}"

# ==========================================
# 3. MCP 핸들러 (툴 3개로 확장)
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
    common_annotations = {
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
        "idempotentHint": True
    }
    
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "analyze_vision_full",
                    "description": f"Provides a comprehensive visual analysis report including object detection, directions, and safety warnings from an image URL. Provided by {SERVICE_NAME}.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string", "description": "Public URL of the image"}},
                        "required": ["image_url"]
                    },
                    "annotations": {"title": "Full Vision Analysis", **common_annotations}
                },
                {
                    "name": "check_forward_danger",
                    "description": f"Checks the image URL specifically for urgent collision hazards or extremely close obstacles in front of the user. Provided by {SERVICE_NAME}.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string", "description": "Public URL of the image"}},
                        "required": ["image_url"]
                    },
                    "annotations": {"title": "Collision Danger Check", **common_annotations}
                },
                {
                    "name": "count_surrounding_objects",
                    "description": f"Counts the number and types of objects present in the image URL to quickly grasp the surroundings. Provided by {SERVICE_NAME}.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string", "description": "Public URL of the image"}},
                        "required": ["image_url"]
                    },
                    "annotations": {"title": "Object Counter", **common_annotations}
                }
            ]
        }
    }

async def handle_tools_call(req_id: str | int, params: dict) -> dict:
    tool_name = params.get("name")
    args = params.get("arguments", {})
    image_url = args.get("image_url", "")

    if not image_url:
        text_content = "⚠️ 오류: 이미지 URL이 입력되지 않았습니다."
    else:
        try:
            if tool_name == "analyze_vision_full":
                text_content = await process_url_async(image_url, mode="full")
            elif tool_name == "check_forward_danger":
                text_content = await process_url_async(image_url, mode="danger")
            elif tool_name == "count_surrounding_objects":
                text_content = await process_url_async(image_url, mode="count")
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
# 4. 엔드포인트 라우팅 (SSE)
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