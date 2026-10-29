def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "terrain_wgs84_proxy.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
