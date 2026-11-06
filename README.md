# PixelForge-AI--Image-ediiting-AI-
PixelForge AI
An AI-assisted image management and editing MVP built with Python, Streamlit, Google Gemini, Pillow, and SQLite.
PixelForge AI lets users upload an image collection, automatically generate image descriptions, search the collection using natural language, apply supported image edits from plain-English instructions, and preserve a version history.

Features
- Image upload: Upload one or multiple PNG, JPG, JPEG, or WEBP images.
- AI image descriptions: Gemini generates concise captions with objects, setting, colors, activities, visual style, and useful search terms.
- Natural-language image search: Search the library with prompts such as “Find beach photos with blue water.”
- AI-assisted editing: Describe an edit in plain English; Gemini converts the instruction into a structured list of supported operations.
- Image transformations: Brightness, contrast, saturation, sharpness, blur, grayscale, sepia, rotation, horizontal/vertical flip, and resize.
- Version history: Save edited versions and restore earlier versions.
- Downloads: Download saved versions as PNG files.
- Dashboard: View image and saved-version counts, plus recent uploads.

Run the Project in Google Colab
The supplied notebook is the recommended way to run this version of PixelForge AI.
1. Clone or download this repository, or download PixelForge_AI_10_Cells.ipynb from GitHub.
2. Open Google Colab.
3. Upload and open the notebook.
4. Run the 10 code cells sequentially, from Cell 1 through Cell 10.
5. Authorize Google Drive access when prompted.
6. Enter your Gemini API key when prompted. The notebook uses getpass, so the key is not written into the notebook source.
7. Wait for Streamlit and the Cloudflare Quick Tunnel to start.
8. Open the trycloudflare.com URL printed by the final cell.
Get a Gemini API key
Create or manage a key through Google AI Studio.

How the AI Workflow Works
1. Image analysis
When an image is uploaded, it is converted to RGB and sent to Gemini with a prompt requesting a factual description. The generated caption and image metadata are stored in SQLite.
2. Natural-language search
The application sends the user's query and the available image IDs, filenames, and captions to Gemini. Gemini returns matching IDs in relevance order. The application maps those IDs back to actual database records and does not accept invented IDs.
3. AI-assisted editing
Gemini interprets the editing instruction and returns structured JSON containing supported operations, unsupported requests, and an explanation. Pillow applies the supported operations to the selected image.
