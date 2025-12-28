from fastapi import FastAPI

# main app, asgi entrypoint
app = FastAPI()

# test endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}
