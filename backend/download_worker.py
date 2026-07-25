"""Detached download worker used so downloads survive dashboard shutdown."""

import asyncio
import json
import sys

from driver_manager import download_driver
from model_manager import download_model, download_openvino_model
from progress import update


async def main() -> None:
    download_id, kind, payload = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

    def report(**kwargs):
        update(download_id, **kwargs)

    try:
        if kind == "model":
            await asyncio.to_thread(
                download_model, payload["repo_id"], payload["filename"], report,
            )
        elif kind == "openvino":
            await asyncio.to_thread(download_openvino_model, payload["repo_id"], report)
        elif kind == "driver":
            await download_driver(payload["tag"], payload["backend"], report)
        else:
            raise ValueError(f"Unknown download kind: {kind}")
        update(download_id, status="done", percent=100, stage="Complete")
    except Exception as exc:
        update(download_id, status="error", percent=0, stage=str(exc))


if __name__ == "__main__":
    asyncio.run(main())
