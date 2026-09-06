"""HanGuard FastAPI 常驻推理服务。

模型在服务启动阶段加载一次，后续请求复用同一个模型实例。为避免同一 GPU
上的并发 generate 导致显存峰值不可控，所有推理请求通过一个异步锁串行执行；
批量请求仍会在锁内按 batch_size 分批推理。
"""

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from infer import CATEGORY_LABELS, build_prompt, infer_batch, load_model

logger = logging.getLogger("hanguard.server")


@dataclass
class Settings:
    model: str = "outputs/hanguard_v5"
    base_model: str | None = None
    batch_size: int = 8
    max_batch_size: int = 32
    max_prompt_chars: int = 20_000
    max_input_len: int = 480
    max_new_tokens: int = 48


settings = Settings()


class ClassifyRequest(BaseModel):
    prompt: str = Field(min_length=1)


class BatchClassifyRequest(BaseModel):
    prompts: list[str] = Field(min_length=1)


class Classification(BaseModel):
    harmful: bool
    harmful_label: str
    category_id: int
    category_label: str


class BatchClassification(BaseModel):
    results: list[Classification]


def _validate_prompts(prompts: list[str]) -> None:
    if len(prompts) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"单次最多提交 {settings.max_batch_size} 条 prompt",
        )
    for index, prompt in enumerate(prompts):
        if not prompt.strip():
            raise HTTPException(status_code=422, detail=f"prompts[{index}] 不能为空")
        if len(prompt) > settings.max_prompt_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"prompts[{index}] 长度为 {len(prompt)}，"
                    f"超过限制 {settings.max_prompt_chars}"
                ),
            )


def _run_inference(app: FastAPI, prompts: list[str]) -> list[Classification]:
    built_prompts = [build_prompt(prompt) for prompt in prompts]
    predictions = infer_batch(
        built_prompts,
        app.state.tokenizer,
        app.state.model,
        batch_size=min(settings.batch_size, len(built_prompts)),
        max_input_len=settings.max_input_len,
        max_new_tokens=settings.max_new_tokens,
        show_progress=False,
    )
    return [
        Classification(
            harmful=harmful_label == "harmful",
            harmful_label=harmful_label,
            category_id=int(category_id),
            category_label=CATEGORY_LABELS[category_id],
        )
        for harmful_label, category_id in predictions
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在加载模型: %s", settings.model)
    tokenizer, model = load_model(settings.model, settings.base_model)
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.inference_lock = asyncio.Lock()
    app.state.ready = True
    logger.info("模型加载完成，服务已就绪")
    try:
        yield
    finally:
        app.state.ready = False
        del app.state.model
        del app.state.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="HanGuard API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="模型尚未就绪")
    return {"status": "ready", "model": settings.model}


async def _classify(request: Request, prompts: list[str]) -> list[Classification]:
    _validate_prompts(prompts)
    async with request.app.state.inference_lock:
        try:
            return await asyncio.to_thread(_run_inference, request.app, prompts)
        except torch.cuda.OutOfMemoryError as exc:
            logger.exception("GPU 显存不足")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="GPU 显存不足，请减小批量") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("推理失败")
            raise HTTPException(status_code=500, detail="模型推理失败") from exc


@app.post("/classify", response_model=Classification)
async def classify(payload: ClassifyRequest, request: Request) -> Classification:
    return (await _classify(request, [payload.prompt]))[0]


@app.post("/classify/batch", response_model=BatchClassification)
async def classify_batch(
    payload: BatchClassifyRequest,
    request: Request,
) -> BatchClassification:
    return BatchClassification(results=await _classify(request, payload.prompts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HanGuard FastAPI 推理服务")
    parser.add_argument("--model", default=settings.model, help="adapter 或完整模型目录")
    parser.add_argument("--base_model", default=None, help="基座模型路径或 HuggingFace ID")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    parser.add_argument("--max_batch_size", type=int, default=settings.max_batch_size)
    parser.add_argument("--max_prompt_chars", type=int, default=settings.max_prompt_chars)
    parser.add_argument("--max_input_len", type=int, default=settings.max_input_len)
    parser.add_argument("--max_new_tokens", type=int, default=settings.max_new_tokens)
    parser.add_argument("--log_level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_batch_size < 1:
        raise ValueError("batch_size 和 max_batch_size 必须大于 0")
    if args.batch_size > args.max_batch_size:
        raise ValueError("batch_size 不能大于 max_batch_size")

    settings.model = args.model
    settings.base_model = args.base_model
    settings.batch_size = args.batch_size
    settings.max_batch_size = args.max_batch_size
    settings.max_prompt_chars = args.max_prompt_chars
    settings.max_input_len = args.max_input_len
    settings.max_new_tokens = args.max_new_tokens

    # 必须保持单 worker；多个 worker 会各自加载一份模型并重复占用显存。
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
