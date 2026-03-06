# app/main.py

from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI(
    title="ML Deployment API",
    description="A simple API for deploying machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the ML Deployment API!", "status": "healthy"}

# GET - Retrieve data, no body required, just query parameters
@app.get("/models")
def list_models():
    return {"models": [ 'classifier_v1', 'regressor_v1', 'sentiment_analyzer_v1' ]}

# POST - Create a new resource, body required
@app.post("/predict")
def predict(data: dict):
    # Placeholder for actual prediction logic
    return {"prediction": "sample_result"}

# PUT - Update an existing resource, body required
@app.put("/models/{model_id}")
def update_model(model_id: str, model_data: dict):
    # Placeholder for actual update logic
    return {"message": f"Model {model_id} updated successfully"}

# DELETE - Remove a resource, no body required
@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    # Placeholder for actual delete logic
    return {"message": f"Model {model_id} deleted successfully"}

# PATCH - Partially update a resource, body required
@app.patch("/models/{model_id}")
def patch_model(model_id: str):
    # Placeholder for actual patch logic
    return {"message": f"Model {model_id} patched successfully"}