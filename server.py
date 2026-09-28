"""hanguard 常驻 HTTP 服务；启动时加载一次模型，串行执行 GPU 推理。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hanguard_model import DEFAULT_MANIFEST
from infer import load_model

logger = logging.getLogger('hanguard.server')


@dataclass
class Settings:
    model: str = str(DEFAULT_MANIFEST)
    device: str = 'cuda:0'
    batch_size: int = 8
    max_batch_size: int = 32
    max_prompt_chars: int = 100_000


settings = Settings()


class ClassifyRequest(BaseModel):
    prompt: str = Field(min_length=1)


class BatchClassifyRequest(BaseModel):
    prompts: list[str] = Field(min_length=1)


class Classification(BaseModel):
    harmful: bool
    harmful_label: str
    safety_label: str
    category_id: int
    category_label: str
    harmful_probability: float
    category_probabilities: dict[str, float]
    binary_threshold: float
    input_tokens: int


class BatchClassification(BaseModel):
    results: list[Classification]


def _validate_prompts(prompts):
    if len(prompts) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f'单次最多提交 {settings.max_batch_size} 条文本')
    for index, prompt in enumerate(prompts):
        if not prompt.strip():
            raise HTTPException(status_code=422, detail=f'第 {index + 1} 条文本不能为空')
        if len(prompt) > settings.max_prompt_chars:
            raise HTTPException(status_code=413, detail=f'第 {index + 1} 条文本超过请求字符上限')


@asynccontextmanager
async def lifespan(app):
    app.state.ready = False
    app.state.classifier = load_model(settings.model, device=settings.device)
    app.state.inference_lock = asyncio.Lock()
    app.state.ready = True
    logger.info('hanguard 已就绪')
    try:
        yield
    finally:
        app.state.ready = False
        app.state.classifier.close()
        del app.state.classifier
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title='hanguard API', lifespan=lifespan)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/ready')
async def ready(request: Request):
    if not getattr(request.app.state, 'ready', False):
        raise HTTPException(status_code=503, detail='模型尚未就绪')
    return {'status': 'ready', 'model': 'hanguard'}


async def _classify(request, prompts):
    _validate_prompts(prompts)
    if not getattr(request.app.state, 'ready', False):
        raise HTTPException(status_code=503, detail='模型尚未就绪')
    async with request.app.state.inference_lock:
        try:
            return await asyncio.to_thread(request.app.state.classifier.predict, prompts, batch_size=settings.batch_size)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            import torch
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail='GPU 显存不足，请减小批量') from error
            logger.exception('推理失败')
            raise HTTPException(status_code=500, detail='模型推理失败') from error


@app.post('/classify', response_model=Classification)
async def classify(payload: ClassifyRequest, request: Request):
    return (await _classify(request, [payload.prompt]))[0]


@app.post('/classify/batch', response_model=BatchClassification)
async def classify_batch(payload: BatchClassifyRequest, request: Request):
    return {'results': await _classify(request, payload.prompts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=settings.model, help='hanguard model.json 清单或所在目录')
    parser.add_argument('--device', default=settings.device)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=settings.batch_size)
    parser.add_argument('--max-batch-size', type=int, default=settings.max_batch_size)
    parser.add_argument('--max-prompt-chars', type=int, default=settings.max_prompt_chars)
    parser.add_argument('--log-level', default='info')
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= args.max_batch_size or args.max_prompt_chars < 1:
        parser.error('批量和字符上限必须为正；batch-size 不得超过 max-batch-size')
    for key in ('model', 'device', 'batch_size', 'max_batch_size', 'max_prompt_chars'):
        setattr(settings, key, getattr(args, key))
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == '__main__':
    main()
