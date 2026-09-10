"""Minimal runtime audit for the coding model's image-input compatibility.

This does not launch Robosuite or any perception/control service. It sends one
synthetic PNG through the same local OpenAI-compatible proxy used by CaP-X and
checks whether the configured coding model can actually consume image_url input.

Usage:
  python research/failure_analysis/audit_qwen_groq_vision.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json

import requests
from PIL import Image

from capx.llm.client import VLM_MODELS


def make_test_image() -> str:
    # Solid green image. The prompt deliberately does not mention the color.
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/qwen/qwen3.8-27b")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    print("=== Qwen/Groq vision compatibility audit ===")
    print(f"model:      {args.model}")
    print(f"server_url: {args.server_url}")
    print(f"in VLM_MODELS: {args.model in VLM_MODELS}")
    print(
        "M2 capture gate currently: "
        + ("OPEN" if args.model in VLM_MODELS else "CLOSED (image feedback will be skipped)")
    )

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the predominant color in this image? Reply with exactly one English color word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": make_test_image()},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    try:
        response = requests.post(args.server_url, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"HTTP REQUEST: FAIL ({type(exc).__name__}: {exc})")
        print("VISION REQUEST ACCEPTED: False")
        return

    print(f"HTTP status: {response.status_code}")
    if response.status_code != 200:
        print("response:")
        print(response.text[:2000])
        print("VISION REQUEST ACCEPTED: False")
        return

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"RESPONSE PARSE: FAIL ({exc})")
        print(response.text[:2000])
        print("VISION REQUEST ACCEPTED: False")
        return

    print(f"model answer: {content!r}")
    answer = str(content).strip().lower().strip(" .,!?:;\"'")
    vision_ok = answer == "green" or "green" in answer.split()
    print(f"VISION REQUEST ACCEPTED: {vision_ok}")

    if vision_ok and args.model not in VLM_MODELS:
        print("NEXT: runtime supports images, but CaP-X VLM_MODELS gating must be patched before M2/M3 use this model.")
    elif vision_ok:
        print("NEXT: model and CaP-X gate both support image feedback; proceed to an M2 smoke trial.")
    else:
        print("NEXT: do not register this model as a VLM yet; inspect provider/proxy vision support first.")


if __name__ == "__main__":
    main()
