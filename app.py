!pip -q install -U streamlit google-genai pillow chromadb sentence-transformers

import os
import sys
import re
import time
import subprocess
from pathlib import Path
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
CHROMA_DIR = BASE / "chroma_db"
DB = BASE / "pixelforge.db"

for folder in [BASE, ASSETS, VERSIONS, CHROMA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app_code = r'''
import os
import io
import re
import json
import uuid
import time
import hashlib
import sqlite3
import logging

from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps
)

from google import genai
from sentence_transformers import SentenceTransformer


BASE = Path("/content/drive/MyDrive/PixelForgeAI")
ASSETS = BASE / "assets"
VERSIONS = BASE / "versions"
CHROMA_DIR = BASE / "chroma_db"
DB = BASE / "pixelforge.db"

MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

for folder in [BASE, ASSETS, VERSIONS, CHROMA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="PixelForge AI",
    page_icon="🎨",
    layout="wide"
)

logging.basicConfig(
    filename=str(BASE / "application.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PixelForge")


@st.cache_resource
def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=key)


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_vector_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="pixelforge_images",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def connect():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize():

    conn = connect()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            current_path TEXT NOT NULL,
            caption TEXT,
            created_at TEXT,
            content_hash TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            description TEXT,
            created_at TEXT,
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    """)


    columns = [
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(images)"
        ).fetchall()
    ]

    if "content_hash" not in columns:
        conn.execute(
            "ALTER TABLE images ADD COLUMN content_hash TEXT"
        )

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_versions_image
        ON versions(image_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_images_created
        ON images(created_at)
    """)

    conn.commit()
    conn.close()


def get_images():

    conn = connect()

    rows = conn.execute("""
        SELECT * FROM images
        ORDER BY created_at DESC
    """).fetchall()

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

    rows = conn.execute("""
        SELECT * FROM versions
        WHERE image_id = ?
        ORDER BY id DESC
    """, (image_id,)).fetchall()

    conn.close()
    return rows


def generate_embedding(text):

    model = get_embedding_model()

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def index_image(image_id, filename, caption):

    collection = get_vector_collection()

    searchable_text = (
        f"Filename: {filename}. "
        f"Image description: {caption}"
    )

    embedding = generate_embedding(searchable_text)

    collection.upsert(
        ids=[image_id],
        embeddings=[embedding],
        documents=[searchable_text],
        metadatas=[{
            "image_id": image_id,
            "filename": filename,
            "caption": str(caption)[:5000]
        }]
    )


def synchronize_embeddings():

    images = get_images()

    if not images:
        return 0

    collection = get_vector_collection()

    indexed = set(
        collection.get()["ids"]
    )

    missing = [
        row for row in images
        if row["id"] not in indexed
    ]

    count = 0

    for row in missing:

        try:
            index_image(
                row["id"],
                row["filename"],
                row["caption"] or row["filename"]
            )
            count += 1

        except Exception as error:
            logger.exception(
                "Embedding synchronization failed: %s",
                row["id"]
            )

    return count


def analyze_image(image):

    client = get_gemini_client()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            """
            Describe this image for a searchable image library.

            Include:
            - Main objects
            - Setting and environment
            - Colors
            - Activities
            - Lighting
            - Visual style
            - Relevant descriptive search keywords

            Be accurate, concise and factual.
            """,
            image
        ]
    )

    return response.text or "No description generated."


def calculate_hash(image_bytes):

    return hashlib.sha256(image_bytes).hexdigest()


def save_upload(uploaded_file):

    raw_bytes = uploaded_file.getvalue()

    content_hash = calculate_hash(raw_bytes)


    conn = connect()

    duplicate = conn.execute("""
        SELECT * FROM images
        WHERE content_hash = ?
    """, (content_hash,)).fetchone()

    conn.close()

    if duplicate:
        return duplicate["id"], False

    image = Image.open(
        io.BytesIO(raw_bytes)
    ).convert("RGB")

    image_id = str(uuid.uuid4())

    filename = Path(uploaded_file.name).name

    file_path = ASSETS / f"{image_id}_original.png"

    image.save(file_path, format="PNG")

    try:
        caption = analyze_image(image)
    except Exception as error:
        logger.exception("Gemini image analysis failed")
        caption = f"Image uploaded: {filename}"

    timestamp = datetime.now().isoformat()

    conn = connect()

    conn.execute("""
        INSERT INTO images
        (id, filename, current_path, caption, created_at, content_hash)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image_id,
        filename,
        str(file_path),
        caption,
        timestamp,
        content_hash
    ))

    conn.execute("""
        INSERT INTO versions
        (image_id, file_path, description, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        image_id,
        str(file_path),
        "Original upload",
        timestamp
    ))

    conn.commit()
    conn.close()


    try:
        index_image(
            image_id,
            filename,
            caption
        )
    except Exception as error:
        logger.exception("Initial embedding failed")

    return image_id, True


def semantic_search(query, top_k=10):

    collection = get_vector_collection()

    total = collection.count()

    if total == 0:
        return []

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=[
            "metadatas",
            "distances",
            "documents"
        ]
    )

    matches = []

    ids = results["ids"][0]
    distances = results["distances"][0]

    for image_id, distance in zip(ids, distances):

        row = get_image(image_id)

        if row is None:
            continue

        similarity = max(
            0.0,
            min(1.0, 1.0 - float(distance))
        )

        matches.append({
            "image": row,
            "similarity": similarity
        })

    return matches


def interpret_edit(image, instruction):

    client = get_gemini_client()

    prompt = f"""
You are an image editing instruction interpreter.

Convert the user's request into supported Pillow operations.

USER REQUEST:
{instruction}

SUPPORTED OPERATIONS:

brightness: value 0.1 to 2.5
contrast: value 0.1 to 2.5
saturation: value 0.0 to 2.5
sharpness: value 0.0 to 3.0
blur: value 0 to 10
grayscale: no value
sepia: no value
rotate: value in degrees
flip_horizontal: no value
flip_vertical: no value
resize: width and height, maximum 3000

Return only JSON:

{{
  "operations": [],
  "unsupported": [],
  "explanation": "Explanation"
}}

Rules:
- Do not invent completed edits.
- Put object insertion, removal, identity changes,
  generative background replacement and other unsupported
  generative edits in the unsupported list.
- Use only the operations listed above.
- Use an empty operations list if nothing is supported.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image],
        config={
            "response_mime_type": "application/json"
        }
    )

    return json.loads(response.text)


def apply_operations(image, operations):

    result = image.copy().convert("RGB")

    allowed = {
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "blur",
        "grayscale",
        "sepia",
        "rotate",
        "flip_horizontal",
        "flip_vertical",
        "resize"
    }

    for operation in operations:

        kind = operation.get("type")
        value = operation.get("value")

        if kind not in allowed:
            continue

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

            width = max(
                1,
                min(int(operation.get("width", result.width)), 3000)
            )

            height = max(
                1,
                min(int(operation.get("height", result.height)), 3000)
            )

            result = result.resize(
                (width, height),
                Image.Resampling.LANCZOS
            )

    return result


def save_edited_version(image_id, image, description):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    file_path = VERSIONS / f"{image_id}_{timestamp}.png"

    image.save(file_path, format="PNG")

    conn = connect()

    conn.execute("""
        UPDATE images
        SET current_path = ?
        WHERE id = ?
    """, (str(file_path), image_id))

    conn.execute("""
        INSERT INTO versions
        (image_id, file_path, description, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        image_id,
        str(file_path),
        description,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def restore_version(image_id, file_path, version_id):

    if not Path(file_path).exists():
        raise FileNotFoundError("Version file does not exist.")

    conn = connect()

    conn.execute("""
        UPDATE images
        SET current_path = ?
        WHERE id = ?
    """, (file_path, image_id))

    conn.execute("""
        INSERT INTO versions
        (image_id, file_path, description, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        image_id,
        file_path,
        f"Restored version {version_id}",
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def safe_image(path):

    try:
        path = Path(path)

        if not path.exists():
            return None

        return Image.open(path).convert("RGB")

    except Exception:
        logger.exception("Could not load image: %s", path)
        return None


initialize()

st.title("🎨 PixelForge AI")

st.caption(
    "AI-powered image management, semantic search "
    "and intelligent image editing workspace"
)


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

    st.caption("Powered by Gemini, ChromaDB and Sentence Transformers")

    st.caption("Week 3 — Semantic Image Search")


images = get_images()

if page in ["Dashboard", "AI Search"]:

    try:
        with st.spinner("Checking image embedding index..."):
            newly_indexed = synchronize_embeddings()

        if newly_indexed:
            st.sidebar.success(
                f"{newly_indexed} images indexed."
            )

    except Exception as error:
        logger.exception("Embedding synchronization failed")
        st.sidebar.warning(
            f"Index synchronization issue: {error}"
        )


if page == "Dashboard":

    st.subheader("Dashboard")

    try:
        conn = connect()

        version_count = conn.execute(
            "SELECT COUNT(*) FROM versions"
        ).fetchone()[0]

        conn.close()

        vector_count = get_vector_collection().count()

    except Exception as error:
        st.error(str(error))
        version_count = 0
        vector_count = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Images", len(images))
    col2.metric("Saved Versions", version_count)
    col3.metric("Indexed Images", vector_count)
    col4.metric("Embedding Model", "MiniLM")

    st.divider()

    st.subheader("Upload Images")

    uploads = st.file_uploader(
        "Upload images to your library",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if st.button("Upload and Analyze", type="primary"):

        if not uploads:

            st.warning("Select at least one image.")

        else:

            progress = st.progress(0)
            status = st.empty()

            successful = 0
            skipped = 0

            for index, uploaded in enumerate(uploads):

                try:

                    status.write(
                        f"Processing {uploaded.name}..."
                    )

                    image_id, created = save_upload(uploaded)

                    if created:
                        successful += 1
                    else:
                        skipped += 1

                except Exception as error:

                    logger.exception(
                        "Upload failed: %s",
                        uploaded.name
                    )

                    st.error(
                        f"{uploaded.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(uploads)
                )

            status.success(
                f"Completed. Uploaded: {successful}. "
                f"Duplicates skipped: {skipped}."
            )

            st.rerun()

    st.divider()

    st.subheader("Recent Images")

    if images:

        columns = st.columns(4)

        for index, row in enumerate(images[:8]):

            with columns[index % 4]:

                image = safe_image(row["current_path"])

                if image:

                    st.image(
                        image,
                        use_container_width=True
                    )

                    st.caption(row["filename"])

    else:

        st.info("Your image library is empty. Upload images to begin.")


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
                        st.image(
                            image,
                            use_container_width=True
                        )

                with col2:

                    st.markdown(f"**{row['filename']}**")

                    st.caption(
                        f"Image ID: {row['id']}"
                    )

                    st.write(row["caption"])

                    st.caption(
                        f"Uploaded: {row['created_at']}"
                    )


elif page == "AI Search":

    st.subheader("🔎 Natural Language Image Search")

    st.write(
        "Search your library using meaning, objects, "
        "scenes, colors and descriptions."
    )

    query = st.text_input(
        "Describe the image you want to find",
        placeholder="Find sunset images with orange skies and mountains"
    )

    top_k = st.slider(
        "Number of search results",
        min_value=1,
        max_value=20,
        value=8
    )

    if st.button("Search Images", type="primary"):

        if not query.strip():

            st.warning("Enter a search query.")

        elif not images:

            st.info("Upload images before searching.")

        else:

            try:

                with st.spinner(
                    "Generating query embedding and searching ChromaDB..."
                ):

                    results = semantic_search(
                        query,
                        top_k
                    )

                    st.session_state["search_results"] = results
                    st.session_state["last_search_query"] = query

            except Exception as error:

                logger.exception("Semantic search failed")
                st.error(f"Search failed: {error}")

    results = st.session_state.get(
        "search_results", []
    )

    if results:

        st.success(
            f"Found {len(results)} semantically relevant images."
        )

        for start in range(0, len(results), 3):

            columns = st.columns(3)

            for index, item in enumerate(
                results[start:start + 3]
            ):

                with columns[index]:

                    row = item["image"]

                    image = safe_image(row["current_path"])

                    if image:

                        st.image(
                            image,
                            use_container_width=True
                        )

                    st.markdown(
                        f"**{row['filename']}**"
                    )

                    st.progress(
                        item["similarity"]
                    )

                    st.caption(
                        f"Similarity: {item['similarity']:.1%}"
                    )

                    with st.expander("Image description"):

                        st.write(row["caption"])

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

                st.image(
                    original,
                    use_container_width=True
                )

            with right:

                st.markdown("**AI editing instructions**")

                instruction = st.text_area(
                    "Describe the edit you want",
                    placeholder=(
                        "Increase brightness and saturation, "
                        "sharpen the image and rotate it 90 degrees."
                    )
                )

                if st.button(
                    "Generate Edit",
                    type="primary"
                ):

                    if not instruction.strip():

                        st.warning("Describe the edit first.")

                    else:

                        try:

                            with st.spinner(
                                "Gemini is interpreting your instructions..."
                            ):

                                plan = interpret_edit(
                                    original,
                                    instruction
                                )

                            operations = plan.get("operations", [])

                            if not isinstance(operations, list):
                                raise ValueError("Invalid editing plan.")

                            edited = apply_operations(
                                original,
                                operations
                            )

                            st.session_state["edit_plan"] = plan
                            st.session_state["edit_image_id"] = image_id
                            st.session_state["edit_instruction"] = instruction
                            st.session_state["edited_preview"] = edited

                        except Exception as error:

                            logger.exception("Image editing failed")
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

                st.write(
                    plan.get("explanation", "")
                )

                if plan.get("operations"):

                    st.write("Applied operations:")

                    st.json(plan["operations"])

                else:

                    st.info("No supported operations were identified.")

                if plan.get("unsupported"):

                    st.warning(
                        "Unsupported requests: "
                        + ", ".join(
                            map(str, plan["unsupported"])
                        )
                    )

                st.image(
                    preview,
                    use_container_width=True
                )

                if plan.get("operations"):

                    if st.button(
                        "Save Edited Image",
                        type="primary"
                    ):

                        try:

                            description = st.session_state.get(
                                "edit_instruction",
                                "Edited image"
                            )

                            save_edited_version(
                                image_id,
                                preview,
                                description
                            )

                            st.success(
                                "New image version saved successfully."
                            )

                            st.session_state.pop(
                                "edit_plan", None
                            )

                            st.session_state.pop(
                                "edited_preview", None
                            )

                            st.rerun()

                        except Exception as error:

                            logger.exception(
                                "Version save failed"
                            )

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

        st.write(
            f"Total versions: {len(versions)}"
        )

        for version in versions:

            with st.container(border=True):

                col1, col2 = st.columns([1, 2])

                image = safe_image(
                    version["file_path"]
                )

                with col1:

                    if image:

                        st.image(
                            image,
                            use_container_width=True
                        )

                with col2:

                    st.markdown(
                        f"**Version {version['id']}**"
                    )

                    st.write(
                        version["description"]
                    )

                    st.caption(
                        version["created_at"]
                    )

                    if st.button(
                        "Restore this version",
                        key=f"restore_{version['id']}"
                    ):

                        try:

                            restore_version(
                                image_id,
                                version["file_path"],
                                version["id"]
                            )

                            st.success(
                                "Version restored successfully."
                            )

                            st.rerun()

                        except Exception as error:

                            st.error(str(error))

                    if image:

                        buffer = io.BytesIO()

                        image.save(
                            buffer,
                            format="PNG"
                        )

                        st.download_button(
                            "Download version",
                            data=buffer.getvalue(),
                            file_name=(
                                f"pixelforge_version_"
                                f"{version['id']}.png"
                            ),
                            mime="image/png",
                            key=f"download_{version['id']}"
                        )


elif page == "About":

    st.subheader("About PixelForge AI")

    st.write("""
    PixelForge AI is an AI-assisted image management and
    editing platform.


    - Upload and manage images.
    - Generate image descriptions using Gemini.
    - Generate semantic embeddings from image captions.
    - Store vectors in persistent ChromaDB.
    - Search images using natural-language queries.
    - Retrieve images using cosine similarity.
    - Interpret natural-language image editing instructions.
    - Apply image transformations using Pillow.
    - Maintain image version history.
    - Restore previous versions.
    - Download edited images.


    - Python
    - Streamlit
    - Google Gemini API
    - Sentence Transformers
    - all-MiniLM-L6-v2
    - ChromaDB
    - SQLite
    - Pillow
    - Google Drive
    - Google Colab


    1. User uploads an image through Streamlit.
    2. Image is stored in Google Drive.
    3. Gemini generates an image caption.
    4. Sentence Transformers generates an embedding.
    5. ChromaDB stores the embedding.
    6. SQLite stores image metadata and version information.
    7. A search query is converted into an embedding.
    8. ChromaDB retrieves similar image vectors.
    9. Streamlit displays matching images.


    PixelForgeAI/
        assets/
        versions/
        chroma_db/
        pixelforge.db
        application.log


    This application supports conventional image transformations.
    Generative object insertion, removal and background replacement
    are not implemented.
    """)

    st.divider()

    st.subheader("System Health")

    col1, col2, col3 = st.columns(3)

    try:

        col1.metric(
            "SQLite",
            "Connected" if DB.exists() else "Not found"
        )

        col2.metric(
            "ChromaDB",
            get_vector_collection().count()
        )

        col3.metric(
            "Google Drive",
            "Mounted" if BASE.exists() else "Not found"
        )

    except Exception as error:

        st.error(str(error))


st.divider()

st.caption(
    "PixelForge AI | Week 3 | "
    "Built using Python, Streamlit, Gemini, ChromaDB and SQLite"
)
'''


app_path = Path("/content/pixelforge_app.py")
app_path.write_text(app_code, encoding="utf-8")

print("PixelForge AI application file created.")


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

print("Starting PixelForge AI...")
time.sleep(10)

if streamlit_process.poll() is not None:
    raise RuntimeError(
        "Streamlit stopped unexpectedly. Check installation and logs."
    )


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
    text=True,
    bufsize=1
)

public_url = None
start_time = time.time()

print("Connecting to Cloudflare...")

while time.time() - start_time < 60:

    line = tunnel_process.stdout.readline()

    if line:
        print(line.strip())

        match = re.search(
            r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com",
            line
        )

        if match:
            public_url = match.group(0)
            break

    if tunnel_process.poll() is not None:
        break


if public_url:

    print("\n" + "=" * 55)
    print("🎨 PIXELFORGE AI IS RUNNING")
    print("=" * 55)
    print("Open your application:")
    print(public_url)
    print("=" * 55)

else:

    print("Cloudflare URL was not detected.")
    print("Check the Cloudflare logs and rerun the tunnel step.")

print("\nStreamlit PID:", streamlit_process.pid)
print("Cloudflare PID:", tunnel_process.pid)
print("Project folder:", BASE)
