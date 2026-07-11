from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Kakao PlayMCP Simple Math Server")

# 보안 요구사항: Origin 검증 (CORS 설정)
# 실제 운영 시에는 playmcp.kakao.com 등 특정 도메인만 허용하는 것이 좋습니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 대회 테스트를 위해 일단 모두 열어둡니다.
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

SERVICE_NAME = "Simple Math Calculator(심플 수학 계산기)"
SUPPORTED_VERSIONS = ["2025-11-25", "2025-03-26"]

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
                        "title": "Addition Tool",
                        "readOnlyHint": "safe",
                        "destructiveHint": "safe",
                        "openWorldHint": "closed",
                        "idempotentHint": "idempotent"
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
                        "title": "Multiplication Tool",
                        "readOnlyHint": "safe",
                        "destructiveHint": "safe",
                        "openWorldHint": "closed",
                        "idempotentHint": "idempotent"
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

@app.post("/mcp")
async def mcp_post_endpoint(request: Request):
    """PlayMCP 클라이언트 요청 처리 (단일 엔드포인트)"""
    # 1. Protocol Version 헤더 검증
    protocol_version = request.headers.get("MCP-Protocol-Version")
    if not protocol_version:
        protocol_version = "2025-03-26" # Fallback
    
    if protocol_version not in SUPPORTED_VERSIONS:
        return JSONResponse({"error": "Unsupported MCP Protocol Version"}, status_code=400)

    # 2. 본문 파싱
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    # 3. 라우팅
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

    return JSONResponse(content=response_data, headers={"Content-Type": "application/json"})

@app.get("/mcp")
async def mcp_get_endpoint():
    """SSE 스트림을 제공하지 않음을 명시 (Stateless)"""
    return Response(status_code=405, content="SSE Stream not supported. Use POST for stateless requests.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)