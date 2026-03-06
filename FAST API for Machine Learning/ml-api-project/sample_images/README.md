# Sample images for testing

Place brain MRI images here to test the prediction API (e.g. with curl or the `/docs` UI).

**Supported formats:** JPEG, PNG, BMP, TIFF  
**Max size per file:** see `MAX_IMAGE_MB` in `.env` (default 5 MB)

Example with curl (from project root):

```bash
curl -X POST "http://localhost:8000/api/v1/predict/tumor" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@sample_images/your_scan.jpg"
```

This directory is kept in Git (via `.gitkeep`) so the folder exists after clone. Add large or private images to `.gitignore` if you don’t want to commit them.
