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

# ✨ [핵심 방어] 프록시 관통 및 카카오 필수 규격 강제 주입 (Chunking 대응 완료)
async def handle_sse(request: Request):
    original_send = request._send
    response_buffer = b""  # 쪼개진 패킷을 모아둘 버퍼

    async def intercepted_send(message: dict):
        nonlocal response_buffer
        
        # 1. 클라우드 프록시 버퍼링 강제 차단 (연결 실패 해결)
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((b"x-accel-buffering", b"no"))
            message["headers"] = headers
            await original_send(message)
            return

        # 2. 본문(Body) 전송 시 청크 조립 및 안전한 파싱
        if message.get("type") == "http.response.body":
            chunk = message.get("body", b"")
            more_body = message.get("more_body", False)
            
            response_buffer += chunk
            messages_to_send = []

            # SSE 규격의 끝인 \n\n 이 버퍼에 완성될 때마다 잘라서 처리
            while b"\n\n" in response_buffer:
                part, response_buffer = response_buffer.split(b"\n\n", 1)
                part_str = part.decode("utf-8", errors="ignore")

                # 이벤트 메시지인 경우에만 파싱 시도
                if part_str.startswith("event: message\ndata: "):
                    try:
                        prefix, json_str = part_str.split("data: ", 1)
                        data = json.loads(json_str)
                        
                        # ✨ Pydantic이 지워버린 annotations 강제 복구
                        if "result" in data and "tools" in data.get("result", {}):
                            for tool in data["result"]["tools"]:
                                tool["annotations"] = {
                                    "title": tool.get("name", "Tool"),
                                    "readOnlyHint": True,
                                    "destructiveHint": False,
                                    "openWorldHint": False,
                                    "idempotentHint": True
                                }
                        
                        # 다시 JSON으로 감싸기
                        new_json_str = json.dumps(data)
                        part = f"{prefix}data: {new_json_str}".encode("utf-8")
                    except Exception as e:
                        # 파싱 실패 시 원본 보존 (에러 무시)
                        pass
                
                messages_to_send.append(part + b"\n\n")

            # 원본 스트림이 끝났는데 버퍼에 남은 데이터가 있다면 밀어내기
            if not more_body and response_buffer:
                messages_to_send.append(response_buffer)
                response_buffer = b""

            # 가공 완료된 메시지들을 안전하게 클라이언트로 전송
            for i, msg_body in enumerate(messages_to_send):
                is_last_chunk = (i == len(messages_to_send) - 1)
                await original_send({
                    "type": "http.response.body",
                    "body": msg_body,
                    # 원본의 상태를 보존 (이 배치의 마지막일 때만 원본 more_body를 따름)
                    "more_body": more_body if is_last_chunk else True
                })

            # 전송할 조립 메시지는 없지만 원본 통신 종료 시그널이 왔을 때
            if not more_body and not messages_to_send:
                await original_send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False
                })
                return

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