import streamlit as st
from io import BytesIO
from openai import OpenAI
from dotenv import dotenv_values
from PIL import Image
import os
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, PointsSelector
import math
import time

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "images_embs"

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

def add_image_to_db(note_text) -> int:
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True)
    new_id = points_count.count + 1
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=new_id,
                vector=calculate_embedding(text=note_text),
                payload={"text": note_text},
            )
        ]
    )
    return new_id

def get_all_notes_from_db():
    qdrant_client = get_qdrant_client()
    notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=1000)[0]
    return [{"id": note.id, "text": note.payload["text"]} for note in notes]

def prepare_image_for_open_ai_from_bytes(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_data}"

def find_relevant_ids_with_openai(query: str, all_notes: list[dict]) -> list[int]:
    prompt = f"Użytkownik wpisał zapytanie: '{query}'. Które z poniższych opisów pasują do tego zapytania? Zwróć tylko ID trafnych obrazków jako listę liczb, np. [1, 4, 7].\n\n"
    prompt += "Opisy obrazków:\n"
    for note in all_notes:
        prompt += f"ID: {note['id']} — {note['text']}\n"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    answer = response.choices[0].message.content
    try:
        ids = eval(answer.strip())
        if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
            return ids
    except:
        pass
    return []

def describe_image(image: Image.Image) -> str:
    base64_image = prepare_image_for_open_ai_from_bytes(image)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Stwórz opis obrazka, jakie widzisz tam elementy?"},
                    {"type": "image_url", "image_url": {"url": base64_image, "detail": "high"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

def calculate_embedding(text: str):
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return result.data[0].embedding

def run_search():
    query_key = f"search_input_{st.session_state['search_input_key']}"
    query = st.session_state.get(query_key, "").strip()

    if not query:
        st.warning("Wpisz coś w pole wyszukiwania.")
        return

    all_notes = get_all_notes_from_db()
    matching_ids = find_relevant_ids_with_openai(query, all_notes)

    found_any = False
    for note in all_notes:
        if note["id"] in matching_ids:
            found_any = True
            with st.container():
                user_path = os.path.join(save_dir, f"{note['id']}.png")
                if os.path.exists(user_path):
                    st.image(user_path, caption=f"Użytkownik {note['id']}", use_container_width=True)
                else:
                    found = False
                    for ext in (".png", ".jpg", ".jpeg"):
                        stock_path = os.path.join("stock_photo", f"{note['id']}{ext}")
                        if os.path.exists(stock_path):
                            st.image(stock_path, caption=f"Stock {note['id']}", use_container_width=True)
                            found = True
                            break
                    if not found:
                        st.warning(f"Brak pliku dla ID {note['id']}")
                with st.expander("📖 Description"):
                    st.markdown(note["text"])
    if not found_any:
        st.info("We didn’t find anything.")

# Inicjalizacja session_state
if "image_text" not in st.session_state:
    st.session_state["image_text"] = ""

if "image_image" not in st.session_state:
    st.session_state["image_image"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "session_ids" not in st.session_state:
    st.session_state["session_ids"] = []

if "search_input_key" not in st.session_state:
    st.session_state["search_input_key"] = 0

if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False

if not st.session_state["splash_shown"]:
    with st.container():
        st.markdown(
            """
            <style>
            .splash-text {
                font-size: 48px;
                font-weight: bold;
                text-align: center;
                margin-top: 200px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="splash-text">MyGallery App</div>', unsafe_allow_html=True)
        time.sleep(2)
        st.session_state["splash_shown"] = True
        st.rerun()

assure_db_collection_exists()
gallery_tab, add_tab, search_tab, reset_tab  = st.tabs(["Gallery","Add photo", "Search", "Reset"])

with add_tab:
    uploaded_file = st.file_uploader("Choose photo", type=["jpg", "jpeg", "png"], key=st.session_state["uploader_key"])
    if uploaded_file is not None:
        st.session_state["image_image"] = Image.open(uploaded_file)
        st.image(st.session_state["image_image"], caption='Wczytany obraz', use_container_width=True)
        st.session_state["image_text"] = describe_image(st.session_state["image_image"])

        if st.button("💾 Save photo"):
            point_id = add_image_to_db(st.session_state["image_text"])
            st.session_state["session_ids"].append(point_id)
            save_path = os.path.join(save_dir, f"{point_id}.png")
            st.session_state["image_image"].save(save_path)
            st.success(f"Photo saved as: {save_path}")

        if st.button("📖 Show description of photo"):
            st.write(st.session_state["image_text"])

with search_tab:
    st.text_input("Search", key=f"search_input_{st.session_state['search_input_key']}")
    if st.button("Search"):
        run_search()
        st.session_state["search_input_key"] += 1
        st.rerun()

with gallery_tab:
    STOCK_DIR = "stock_photo"
    stock_files = []
    if os.path.isdir(STOCK_DIR):
        stock_files = [
            os.path.join(STOCK_DIR, f)
            for f in os.listdir(STOCK_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    user_files = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    all_images = stock_files + user_files

    if not all_images:
        st.info("Brak żadnych obrazów w galerii.")
    else:
        images_per_row = 2
        rows = math.ceil(len(all_images) / images_per_row)
        for i in range(rows):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                idx = i * images_per_row + j
                if idx < len(all_images):
                    img_path = all_images[idx]
                    filename = os.path.basename(img_path)
                    with open(img_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    cols[j].image(img_bytes, caption=filename, use_container_width=True)

with reset_tab:
    if st.button("🔄 Restart MyGallery App"):
        for fname in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, fname))

        qdrant_client = get_qdrant_client()
        if st.session_state["session_ids"]:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=st.session_state["session_ids"],
                wait=True
            )

        st.session_state.clear()
        st.session_state["uploader_key"] = 1
        st.session_state["session_ids"] = []
        st.session_state["search_input_key"] = 0
        st.success("Data has been reset")
        st.rerun()
