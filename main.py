from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uuid

app = FastAPI(title="Simple Math Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SERVICE_NAME = "Simple Math Calculator(심플 수학 계산기)"
SUPPORTED_VERSIONS = ["2025-11-25", "2025-03-26"]

# 세션 관리를 위한 전역 딕셔너리 (Session ID -> asyncio.Queue)
active_sessions = {}

def handle_initialize(req_id: str | int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "simple-math-mcp",
                "version": "1.0.0"
            }
        }
    }

def handle_tools_list(req_id: str | int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "calculate_add",
                    "description": f"Adds two numbers. (두 숫자를 더합니다) Provided by {SERVICE_NAME}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": {
                        "title": "Addition Tool",  # 타이틀은 문자열 그대로 둡니다 (툴에 맞게 변경)
                        "readOnlyHint": True,      # 데이터를 읽기만 하거나 상태 변경이 없으므로 True
                        "destructiveHint": False,  # 데이터 삭제/수정 등 파괴적인 작업이 아니므로 False
                        "openWorldHint": False,    # 외부 API/웹과 상호작용하는 오픈월드가 아니므로 False
                        "idempotentHint": True     # 같은 입력에 항상 같은 결과가 보장되므로 True
                    }
                },
                {
                    "name": "calculate_multiply",
                    "description": f"Multiplies two numbers. (두 숫자를 곱합니다) Provided by {SERVICE_NAME}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": {
                        "title": "Addition Tool",  # 타이틀은 문자열 그대로 둡니다 (툴에 맞게 변경)
                        "readOnlyHint": True,      # 데이터를 읽기만 하거나 상태 변경이 없으므로 True
                        "destructiveHint": False,  # 데이터 삭제/수정 등 파괴적인 작업이 아니므로 False
                        "openWorldHint": False,    # 외부 API/웹과 상호작용하는 오픈월드가 아니므로 False
                        "idempotentHint": True     # 같은 입력에 항상 같은 결과가 보장되므로 True
                    }
                },
                {
                    "name": "calculate_power",
                    "description": f"Calculates the power of a number. (첫 번째 숫자를 두 번째 숫자만큼 거듭제곱합니다) Provided by {SERVICE_NAME}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "Base number (밑)"},
                            "b": {"type": "number", "description": "Exponent (지수)"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": {
                        "title": "Addition Tool",  # 타이틀은 문자열 그대로 둡니다 (툴에 맞게 변경)
                        "readOnlyHint": True,      # 데이터를 읽기만 하거나 상태 변경이 없으므로 True
                        "destructiveHint": False,  # 데이터 삭제/수정 등 파괴적인 작업이 아니므로 False
                        "openWorldHint": False,    # 외부 API/웹과 상호작용하는 오픈월드가 아니므로 False
                        "idempotentHint": True     # 같은 입력에 항상 같은 결과가 보장되므로 True
                    }
                }
            ]
        }
    }

def handle_tools_call(req_id: str | int, params: dict) -> dict:
    tool_name = params.get("name")
    args = params.get("arguments", {})
    a = args.get("a", 0)
    b = args.get("b", 0)

    try:
        if tool_name == "calculate_add":
            result_val = a + b
            text_content = f"### 덧셈 결과\n**{a}** + **{b}** = **{result_val}**"
        elif tool_name == "calculate_multiply":
            result_val = a * b
            text_content = f"### 곱셈 결과\n**{a}** × **{b}** = **{result_val}**"
        elif tool_name == "calculate_power":
            result_val = a ** b
            text_content = f"### 거듭제곱 결과\n**{a}**^**{b}** = **{result_val}**"
        else:
            text_content = f"Error: 알 수 없는 툴 이름입니다. ({tool_name})"
    except Exception as e:
        text_content = f"Error: 계산 중 오류가 발생했습니다. ({str(e)})"

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": text_content
                }
            ]
        }
    }

@app.get("/mcp")
async def mcp_get_endpoint(request: Request):
    """MCP 클라이언트 연결 수락 및 SSE 스트림 반환"""
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    active_sessions[session_id] = queue

    async def event_generator():
        # 클라이언트에게 메시지를 보낼 POST 엔드포인트를 세션 ID와 함께 알려줌
        yield f"event: endpoint\ndata: /mcp?sessionId={session_id}\n\n"
        
        try:
            while True:
                try:
                    # 15초 동안 큐에 데이터가 들어오길 대기
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    # 데이터가 들어오면 클라이언트에게 JSON 형태로 전송
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # 15초 동안 메시지가 없으면 연결 유지를 위해 keepalive 전송
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # 클라이언트 연결 종료 시 세션 정리
            if session_id in active_sessions:
                del active_sessions[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp")
async def mcp_post_endpoint(request: Request, sessionId: str = None):
    """클라이언트로부터의 JSON-RPC 요청 수신"""
    
    # 본문 파싱
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    # 라우팅 및 응답 데이터 생성
    if method == "initialize":
        response_data = handle_initialize(req_id)
    elif method == "tools/list":
        response_data = handle_tools_list(req_id)
    elif method == "tools/call":
        response_data = handle_tools_call(req_id, params)
    elif method == "notifications/initialized":
        return Response(status_code=202)
    else:
        response_data = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    # 세션 ID가 존재하고 유효한 경우 큐에 응답 데이터를 넣고 202 Accepted 반환
    if sessionId and sessionId in active_sessions:
        await active_sessions[sessionId].put(response_data)
        return Response(status_code=202) # 타임아웃 방지의 핵심
    else:
        # SSE 연결 없이 단발성 POST 테스트를 하는 경우를 위한 Fallback
        return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)