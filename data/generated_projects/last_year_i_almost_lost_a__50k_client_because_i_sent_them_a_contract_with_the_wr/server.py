import uvicorn
from fastapi import FastAPI, Depends
dependencies = []
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Workflow Reliability Console API'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)