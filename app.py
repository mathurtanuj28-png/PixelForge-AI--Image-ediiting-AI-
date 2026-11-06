!pip -q install -U streamlit google-genai pillow

import os
import sys
import json
import uuid
import time
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from getpass import getpass

from google.colab import drive

drive.mount("/content/drive")

api_key = getpass("Enter your Gemini API key: ").strip()
if not api_key:
    raise ValueError("Gemini API key is required.")

os.environ["GEMINI_API_KEY"] = api_key

BASE = Path("/content/drive/MyDrive/PixelForgeAI")
ASSETS = BASE / "assets"
VERSIONS = BASE / "versions"
DB = BASE / "pixelforge.db"

ASSETS.mkdir(parents=True, exist_ok=True)
VERSIONS.mkdir(parents=True, exist_ok=True)

app_code = r'''
import os
import json
import uuid
import sqlite3
import re
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from google import genai

BASE = Path("/content/drive/MyDrive/PixelForgeAI")
ASSETS = BASE / "assets"
VERSIONS = BASE / "versions"
DB = BASE / "pixelforge.db"
MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

st.set_page_config(
    page_title="PixelForge AI",
    page_icon="🎨",
    layout="wide"
)

def connect():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def initialize():
    conn = connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT,
            current_path TEXT,
            caption TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            file_path TEXT,
            description TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_images():
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM images ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows

def get_image(image_id):
    conn = connect()
    row = conn.execute(
        "SELECT * FROM images WHERE id = ?",
        (image_id,)
    ).fetchone()
    conn.close()
    return row

def get_versions(image_id):
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM versions WHERE image_id = ? ORDER BY id DESC",
        (image_id,)
    ).fetchall()
    conn.close()
    return rows

def save_upload(uploaded_file):
    image_id = str(uuid.uuid4())
    filename = Path(uploaded_file.name).name
    image = Image.open(uploaded_file).convert("RGB")
    file_path = ASSETS / f"{image_id}_original.png"
    image.save(file_path, format="PNG")

    caption = analyze_image(image)

    conn = connect()
    conn.execute(
        "INSERT INTO images VALUES (?, ?, ?, ?, ?)",
        (
            image_id,
            filename,
            str(file_path),
            caption,
            datetime.now().isoformat()
        )
    )
    conn.execute(
        "INSERT INTO versions (image_id, file_path, description, created_at) VALUES (?, ?, ?, ?)",
        (
            image_id,
            str(file_path),
            "Original upload",
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

def analyze_image(image):
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            "Describe this image in detail for an image library. "
            "Include the main objects, people if present, setting, "
            "colors, activities, visual style and useful search keywords. "
            "Keep the description concise and factual.",
            image
        ]
    )
    return response.text or "No description generated."

def search_images(query, images):
    if not images:
        return []

    library = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "caption": row["caption"]
        }
        for row in images
    ]

    prompt = f"""
You are an image library search assistant.

User query:
{query}

Available images:
{json.dumps(library, ensure_ascii=False)}

Return only valid JSON:
{{"matching_ids": ["image_id_1", "image_id_2"]}}

Return matching image IDs in relevance order.
Do not invent IDs. Return an empty list if nothing matches.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    try:
        result = json.loads(response.text)
        ids = result.get("matching_ids", [])
        lookup = {row["id"]: row for row in images}
        return [lookup[i] for i in ids if i in lookup]
    except Exception:
        terms = query.lower().split()
        return [
            row for row in images
            if any(
                term in (str(row["caption"]) + " " + str(row["filename"])).lower()
                for term in terms
            )
        ]

def interpret_edit(image, instruction):
    prompt = f"""
You are an image editing assistant.

Analyze the user's requested edit and convert it into supported operations.

User instruction:
{instruction}

Only use these operations:
brightness: value from 0.1 to 2.5, where 1 is unchanged
contrast: value from 0.1 to 2.5, where 1 is unchanged
saturation: value from 0.0 to 2.5, where 1 is unchanged
sharpness: value from 0.0 to 3.0, where 1 is unchanged
blur: value from 0 to 10
grayscale: no value required
sepia: no value required
rotate: value in degrees
flip_horizontal: no value required
flip_vertical: no value required
resize: width and height, maximum 3000 each

Return only valid JSON in this format:
{{
  "operations": [
    {{"type": "brightness", "value": 1.2}}
  ],
  "unsupported": [],
  "explanation": "Short explanation"
}}

If the user requests adding objects, removing objects, changing a person's identity,
generative background replacement, or any edit that cannot be achieved using
the listed operations, put that request in unsupported.
Do not pretend unsupported operations were completed.
Use an empty operations list if no supported operation is requested.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image],
        config={"response_mime_type": "application/json"}
    )

    return json.loads(response.text)

def apply_operations(image, operations):
    result = image.copy().convert("RGB")

    for operation in operations:
        kind = operation.get("type")
        value = operation.get("value")

        if kind == "brightness":
            result = ImageEnhance.Brightness(result).enhance(
                max(0.1, min(float(value), 2.5))
            )

        elif kind == "contrast":
            result = ImageEnhance.Contrast(result).enhance(
                max(0.1, min(float(value), 2.5))
            )

        elif kind == "saturation":
            result = ImageEnhance.Color(result).enhance(
                max(0.0, min(float(value), 2.5))
            )

        elif kind == "sharpness":
            result = ImageEnhance.Sharpness(result).enhance(
                max(0.0, min(float(value), 3.0))
            )

        elif kind == "blur":
            result = result.filter(
                ImageFilter.GaussianBlur(
                    radius=max(0, min(float(value), 10))
                )
            )

        elif kind == "grayscale":
            result = ImageOps.grayscale(result).convert("RGB")

        elif kind == "sepia":
            gray = ImageOps.grayscale(result)
            result = ImageOps.colorize(
                gray,
                black="#34251A",
                white="#F5D9A6"
            ).convert("RGB")

        elif kind == "rotate":
            result = result.rotate(
                float(value),
                expand=True,
                fillcolor="white"
            )

        elif kind == "flip_horizontal":
            result = ImageOps.mirror(result)

        elif kind == "flip_vertical":
            result = ImageOps.flip(result)

        elif kind == "resize":
            width = max(1, min(int(operation.get("width", result.width)), 3000))
            height = max(1, min(int(operation.get("height", result.height)), 3000))
            result = result.resize((width, height), Image.Resampling.LANCZOS)

    return result

def save_edited_version(image_id, image, description):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = VERSIONS / f"{image_id}_{timestamp}.png"
    image.save(file_path, format="PNG")

    conn = connect()
    conn.execute(
        "UPDATE images SET current_path = ? WHERE id = ?",
        (str(file_path), image_id)
    )
    conn.execute(
        "INSERT INTO versions (image_id, file_path, description, created_at) VALUES (?, ?, ?, ?)",
        (
            image_id,
            str(file_path),
            description,
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

def restore_version(image_id, file_path):
    conn = connect()
    conn.execute(
        "UPDATE images SET current_path = ? WHERE id = ?",
        (file_path, image_id)
    )
    conn.commit()
    conn.close()

def safe_image(path):
    path = Path(path)
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")

initialize()

st.title("🎨 PixelForge AI")
st.caption("AI-powered image library, visual search and image editing workspace")

with st.sidebar:
    st.title("PixelForge")
    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "Image Library",
            "AI Search",
            "AI Editor",
            "Version History",
            "About"
        ]
    )

    st.divider()
    st.caption("Powered by Gemini and Pillow")

images = get_images()

if page == "Dashboard":
    st.subheader("Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", len(images))

    conn = connect()
    version_count = conn.execute(
        "SELECT COUNT(*) FROM versions"
    ).fetchone()[0]
    conn.close()

    col2.metric("Saved Versions", version_count)
    col3.metric("AI Model", "Gemini Flash-Lite")

    st.divider()

    st.subheader("Upload Images")

    uploads = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if st.button("Upload and Analyze", type="primary"):
        if not uploads:
            st.warning("Select at least one image.")
        else:
            progress = st.progress(0)
            status = st.empty()

            for index, uploaded in enumerate(uploads):
                try:
                    status.write(f"Analyzing {uploaded.name}...")
                    save_upload(uploaded)
                except Exception as error:
                    st.error(f"{uploaded.name}: {error}")

                progress.progress((index + 1) / len(uploads))

            status.success("Upload process completed.")
            st.rerun()

    st.divider()
    st.subheader("Recent Images")

    if images:
        columns = st.columns(4)
        for index, row in enumerate(images[:8]):
            with columns[index % 4]:
                image = safe_image(row["current_path"])
                if image:
                    st.image(image, use_container_width=True)
                    st.caption(row["filename"])
        st.button("Refresh dashboard", on_click=st.rerun)
    else:
        st.info("Your image library is empty. Upload images to get started.")

elif page == "Image Library":
    st.subheader("Image Library")

    if not images:
        st.info("No images uploaded yet.")
    else:
        for row in images:
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    image = safe_image(row["current_path"])
                    if image:
                        st.image(image, use_container_width=True)

                with col2:
                    st.markdown(f"**{row['filename']}**")
                    st.caption(f"Image ID: {row['id']}")
                    st.write(row["caption"])
                    st.caption(f"Uploaded: {row['created_at']}")

elif page == "AI Search":
    st.subheader("Natural Language Image Search")

    query = st.text_input(
        "Describe the image you want to find",
        placeholder="Find beach photos with blue water"
    )

    if st.button("Search Images", type="primary"):
        if not query.strip():
            st.warning("Enter a search query.")
        elif not images:
            st.info("Upload images before searching.")
        else:
            with st.spinner("Gemini is searching your library..."):
                try:
                    results = search_images(query, images)
                    st.session_state["search_results"] = results
                except Exception as error:
                    st.error(str(error))

    results = st.session_state.get("search_results", [])

    if results:
        st.success(f"Found {len(results)} matching images.")
        columns = st.columns(3)

        for index, row in enumerate(results):
            with columns[index % 3]:
                image = safe_image(row["current_path"])
                if image:
                    st.image(image, use_container_width=True)
                    st.markdown(f"**{row['filename']}**")
                    st.caption(row["caption"])

    elif "search_results" in st.session_state:
        st.info("No matching images found.")

elif page == "AI Editor":
    st.subheader("AI Image Editor")

    if not images:
        st.info("Upload an image before opening the editor.")
    else:
        choices = {
            f"{row['filename']} | {row['id'][:8]}": row["id"]
            for row in images
        }

        selected_label = st.selectbox(
            "Choose an image",
            list(choices.keys())
        )

        image_id = choices[selected_label]
        row = get_image(image_id)
        original = safe_image(row["current_path"])

        if original:
            left, right = st.columns(2)

            with left:
                st.markdown("**Current image**")
                st.image(original, use_container_width=True)

            with right:
                st.markdown("**AI editing instructions**")
                instruction = st.text_area(
                    "Describe the edit you want",
                    placeholder="Increase brightness and saturation, sharpen the image, and rotate it 90 degrees."
                )

                if st.button("Generate Edit", type="primary"):
                    if not instruction.strip():
                        st.warning("Describe the edit first.")
                    else:
                        try:
                            with st.spinner("Gemini is interpreting your instructions..."):
                                plan = interpret_edit(original, instruction)

                            st.session_state["edit_plan"] = plan
                            st.session_state["edit_image_id"] = image_id
                            st.session_state["edit_instruction"] = instruction

                            edited = apply_operations(
                                original,
                                plan.get("operations", [])
                            )

                            st.session_state["edited_preview"] = edited

                        except Exception as error:
                            st.error(f"Editing failed: {error}")

            plan = st.session_state.get("edit_plan")
            preview = st.session_state.get("edited_preview")

            if (
                plan
                and preview is not None
                and st.session_state.get("edit_image_id") == image_id
            ):
                st.divider()
                st.subheader("Edit Preview")

                if plan.get("explanation"):
                    st.write(plan["explanation"])

                if plan.get("operations"):
                    st.write("Applied operations:")
                    st.json(plan["operations"])
                else:
                    st.info("No supported image operations were identified.")

                if plan.get("unsupported"):
                    st.warning(
                        "These requested edits cannot be performed by this editor: "
                        + ", ".join(map(str, plan["unsupported"]))
                    )

                st.image(preview, use_container_width=True)

                if plan.get("operations"):
                    if st.button("Save Edited Image", type="primary"):
                        try:
                            description = st.session_state.get(
                                "edit_instruction",
                                "Edited image"
                            )
                            save_edited_version(image_id, preview, description)
                            st.success("New image version saved.")
                            st.session_state.pop("edit_plan", None)
                            st.session_state.pop("edited_preview", None)
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))

elif page == "Version History":
    st.subheader("Image Version History")

    if not images:
        st.info("No images available.")
    else:
        choices = {
            f"{row['filename']} | {row['id'][:8]}": row["id"]
            for row in images
        }

        selected_label = st.selectbox(
            "Select an image",
            list(choices.keys())
        )

        image_id = choices[selected_label]
        versions = get_versions(image_id)

        st.write(f"Total versions: {len(versions)}")

        for version in versions:
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    image = safe_image(version["file_path"])
                    if image:
                        st.image(image, use_container_width=True)

                with col2:
                    st.markdown(f"**Version {version['id']}**")
                    st.write(version["description"])
                    st.caption(version["created_at"])

                    if st.button(
                        "Restore this version",
                        key=f"restore_{version['id']}"
                    ):
                        restore_version(image_id, version["file_path"])
                        st.success("Version restored.")
                        st.rerun()

                    if image:
                        from io import BytesIO
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        st.download_button(
                            "Download version",
                            data=buffer.getvalue(),
                            file_name=f"pixelforge_version_{version['id']}.png",
                            mime="image/png",
                            key=f"download_{version['id']}"
                        )

elif page == "About":
    st.subheader("About PixelForge AI")

    st.write("""
PixelForge AI is an AI-assisted image management and editing MVP.

### Features
- Upload and manage an image library
- Generate image descriptions using Gemini
- Search images using natural language
- Convert natural-language instructions into image editing operations
- Apply real image transformations
- Save and restore image versions
- Download edited images

### Technology Stack
- Python
- Streamlit
- Google Gemini API
- Pillow
- SQLite
- Google Drive

### Current editing limitations
This MVP supports conventional image transformations. Generative object removal,
object insertion and background replacement are not implemented.
    """)

st.divider()
st.caption("PixelForge AI | Built with Python, Streamlit and Gemini")
'''

app_path = Path("/content/pixelforge_app.py")
app_path.write_text(app_code)

cloudflared_path = "/content/cloudflared"

if not Path(cloudflared_path).exists():
    subprocess.run(
        [
            "wget",
            "-q",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O",
            cloudflared_path
        ],
        check=True
    )
    os.chmod(cloudflared_path, 0o755)

streamlit_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(8)

tunnel_process = subprocess.Popen(
    [
        cloudflared_path,
        "tunnel",
        "--url",
        "http://localhost:8501",
        "--no-autoupdate"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

public_url = None
start_time = time.time()

while time.time() - start_time < 45:
    line = tunnel_process.stdout.readline()
    if line:
        print(line.strip())
        if "trycloudflare.com" in line:
            import re
            match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break

if public_url:
    print("\nPIXELFORGE AI IS RUNNING")
    print("Open this URL:", public_url)
else:
    print("The tunnel URL was not detected.")
    print("Check the Cloudflare output above and rerun the final tunnel step if needed.")
