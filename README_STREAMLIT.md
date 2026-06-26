# VERIFYGN Streamlit Detection App

Separate Streamlit deployment bundle for image detection with:

- upload image mode
- browser camera mode
- system webcam fallback mode
- VERIFYGNCLOUD workflow inference by default
- prompt/class mappings controlled from script-side config
- class count result per image
- detected image preview with bounding boxes/labels
- hidden Streamlit default chrome
- configurable company logo and heading

## Folder layout

```text
sam3_streamlit_app/
  app.py
  backend.py
  config.py
  requirements.txt
  requirements_streamlit.txt
  assets/VFN_logo.png
```

`model/` is optional now. The default backend is VERIFYGNCLOUD, so Streamlit Cloud does not need to download or load `sam3.pt`.

## VERIFYGNCLOUD Setup

The app is configured in `config.py` to use:

```python
INFERENCE_BACKEND = "verifygncloud"
VERIFYGNCLOUD_API_URL = "https://serverless.roboflow.com"
VERIFYGNCLOUD_WORKSPACE = "rajkumarm"
VERIFYGNCLOUD_WORKFLOW_ID = "general-segmentation-api"
VERIFYGNCLOUD_IMAGE_INPUT = "image"
VERIFYGNCLOUD_CLASSES_INPUT = "classes"
VERIFYGNCLOUD_ANNOTATED_OUTPUT = "annotated_image"
VERIFYGNCLOUD_PREDICTIONS_OUTPUT = "predictions"
```

Add your API key in Streamlit Cloud:

```toml
VERIFYGNCLOUD_API_KEY = "your_verifygncloud_api_key_here"
```

The app also accepts these fallback formats if needed:

```toml
ROBOFLOW_API_KEY = "your_verifygncloud_api_key_here"
```

or:

```toml
[verifygncloud]
api_key = "your_verifygncloud_api_key_here"
```

For localhost testing, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` inside this app folder and paste the real key there:

```text
SAM/sam3_streamlit_app/.streamlit/secrets.toml
```

In Streamlit Cloud, open:

```text
Manage app -> Settings -> Secrets
```

Then paste the key, save, and reboot the app.

As a last-resort deployment fallback, you can also set `VERIFYGNCLOUD_API_KEY` directly in `config.py`, but avoid that for public GitHub repositories.

## Prompt and Class Mapping

Edit `PROMPT_CLASS_MAP` in `config.py`:

```python
PROMPT_CLASS_MAP = [
    ("GOLD", "INSERT_NUT"),
]
```

The first value is sent to the VERIFYGNCLOUD workflow as the prompt/class input. The second value is the class name shown in the UI count table.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If your local environment already had an older `inference-sdk` installed, refresh it with:

```bash
pip install --upgrade "inference-sdk>=1.3.2"
```

You can also run:

```bash
python app.py
```

The script will auto-launch itself with Streamlit.

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub.
2. Make sure these files are in GitHub:
   - `app.py`
   - `backend.py`
   - `config.py`
   - `requirements.txt`
   - `assets/VFN_logo.png`
3. In Streamlit Community Cloud, create the app from your GitHub repo.
4. Set the main file path to:

```text
app.py
```

If this folder is inside a bigger repo, use:

```text
SAM/sam3_streamlit_app/app.py
```

5. Add `VERIFYGNCLOUD_API_KEY` in Streamlit Secrets.
6. Reboot the app.

## Optional Local Model Mode

If you want to use a local model instead of VERIFYGNCLOUD, set this in `config.py`:

```python
INFERENCE_BACKEND = "local"
```

Then configure `MODEL_PATH`, `MODEL_DOWNLOAD_URL`, or Hugging Face model settings in `config.py`.

Local model mode also needs model packages such as `torch`, `ultralytics`, `timm`, `transformers`, and related dependencies. VERIFYGNCLOUD mode does not need them.

For Streamlit Community Cloud, API mode is recommended because very large model files can cause long downloads, memory errors, or app crashes.

## Notes

- Confidence and IoU are script-side only and hidden from the UI.
- Branding is controlled through `config.py` and `assets/VFN_logo.png`.
- The UI is intentionally limited to logo, heading, input source, Detect, Reset, raw image, detected image, and class/count output.
- If the browser camera stays blank, allow browser and Windows camera permissions or use the `System Webcam` input mode.
