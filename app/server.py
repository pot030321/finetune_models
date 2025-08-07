import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.rag_chain import final_chain

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, final_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("FASTAPI_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_SERVER_PORT", 6868)),
    )
