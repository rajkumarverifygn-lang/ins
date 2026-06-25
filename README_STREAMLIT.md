# SAM3 Streamlit App

Separate Streamlit deployment bundle for SAM3 text-guided detection with:

- upload image mode
- browser camera mode
- system webcam fallback mode
- prompt/class mappings controlled from script-side config
- class count result per image
- bounding box preview per image
- hidden Streamlit default chrome
- configurable company logo and heading

## Folder layout

```text
sam3_streamlit_app/
  app.py
  backend.py
  config.py
  requirements_streamlit.txt
  assets/VFN_logo.png
  model/sam3.pt
```

## Setup

1. Create or activate your Python environment.
2. Install PyTorch for your machine.
3. Install the app requirements:

```bash
pip install -r requirements.txt
```

4. Put your SAM3 checkpoint at:

```text
model/sam3.pt
```

5. Optional:
   - Edit `config.py` to change the logo path, title, subtitle, hardcoded prompt/class mapping, confidence, IoU, and defaults.

## Run

```bash
streamlit run app.py
```

On Windows you can also use `RUN_STREAMLIT_WINDOWS.bat`.

You can also run:

```bash
python app.py
```

The script will auto-launch itself with Streamlit.

## Deploy to Streamlit Community Cloud

1. Push the app folder and required files to GitHub.
2. Make sure these files are in GitHub:
   - `app.py`
   - `backend.py`
   - `config.py`
   - `requirements.txt`
   - `assets/VFN_logo.png`
3. If `model/sam3.pt` is small enough for GitHub, push it too.
4. If `model/sam3.pt` is too large, use Git LFS or change the app to download the model at runtime from cloud storage.
5. In Streamlit Community Cloud, click `Create app`.
6. Choose your GitHub repo, branch, and entry file:

```text
SAM/sam3_streamlit_app/app.py
```

## Important deployment note

Streamlit Community Cloud can only access files that are available in the deployed environment. That means:

- local files on your PC are not available unless they are uploaded to GitHub or downloaded by the app at runtime
- your `assets` folder should be uploaded to GitHub
- your `model` folder should also be uploaded if you want the app to load `model/sam3.pt` directly

## Prompt and class mapping

Edit `PROMPT_CLASS_MAP` in `config.py` with rows like:

```python
PROMPT_CLASS_MAP = [
    ("person wearing helmet", "helmet_person"),
    ("red apple", "apple"),
    ("car", "vehicle"),
]
```

The `Prompt` is sent to SAM3. The `Class` is what appears on the UI and count table.

## Notes

- Confidence is hardcoded in `config.py` and intentionally hidden from the UI.
- Bounding boxes are shown on output images.
- Confidence values are not shown on the UI.
- Branding is controlled through `config.py` and `assets/VFN_logo.png`.
- Model path, prompt/class mapping, confidence, and IoU are script-side only and not shown on the UI.
- The UI is intentionally limited to logo, heading, input source, Detect, Reset, raw image, detected image, and class/count output.
- If the browser camera stays blank, allow browser and Windows camera permissions or use the `System Webcam` input mode.
