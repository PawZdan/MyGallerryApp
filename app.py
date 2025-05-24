import streamlit as st
from io import BytesIO
from openai import OpenAI
from dotenv import dotenv_values, load_dotenv
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
#############
############# * v4 - wyszukiwanie zdjęć na podstawie opisu. Dopisz teraz zapisywanie obrazkow do folderu,
# ############### i kasowanie przy ponownym uruchomieniu aplikacji dodaj pawle to co wyswietlasz do session state
#############
#env = dotenv_values(".env")
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

load_dotenv()


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
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
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    new_id = points_count.count + 1
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=new_id,
                vector=calculate_embedding(text=note_text),
                payload={
                    "text": note_text,
                },
            )
        ]
    )
    return new_id

def get_all_notes_from_db():
    qdrant_client = get_qdrant_client()
    notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=1000)[0]
    result = []
    for note in notes:
        result.append({
            "id": note.id,
            "text": note.payload["text"],
        })
    return result


def prepare_image_for_open_ai_from_bytes(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Możesz też użyć JPEG jeśli wolisz
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
        ids = eval(answer.strip())  # np. [1, 3, 7]
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
                    {
                        "type": "text",
                        "text": "Stwórz opis obrazka, jakie widzisz tam elementy?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

def calculate_embedding(text: str):
    result = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-large",
    )
    return result.data[0].embedding

################### Session state initialization  ########################
##if "note_image_md5" not in st.session_state:
#    st.session_state["note_image_md5"] = None

#if "image_bytes" not in st.session_state:
#    st.session_state["image_bytes"] = None

if "image_text" not in st.session_state:
    st.session_state["image_text"] = ""

if "image_image" not in st.session_state:
    st.session_state["image_image"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "session_ids" not in st.session_state:
    st.session_state["session_ids"] = []

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False


if "active_tab_radio" not in st.session_state:
    st.session_state["active_tab_radio"] = "Gallery"

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
        time.sleep(2)  # czas trwania splash screena
        st.session_state["splash_shown"] = True
        st.rerun()



assure_db_collection_exists()
tab = st.radio(
    "",
    options=["Gallery", "Add photo", "Search", "Reset"],
    key="active_tab_radio",  # to automatycznie zapisuje wybór do session_state
    horizontal=True
)
#st.session_state["active_tab"] = tab
if tab == "Gallery":
    # ========== GALLERY ==========
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
        num_images = len(all_images)
        rows = math.ceil(num_images / images_per_row)

        for i in range(rows):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                idx = i * images_per_row + j
                if idx < num_images:
                    img_path = all_images[idx]
                    filename = os.path.basename(img_path)
                    with open(img_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    cols[j].image(img_bytes, caption=filename, use_container_width=True)

elif tab == "Add photo":
    # ========== ADD ==========
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

elif tab == "Search":
    # ========== SEARCH ==========
    query = st.text_input(
        "Search",
        value=st.session_state.get("search_query", ""),
        key="search_query_input")
    st.session_state["search_query"] = query
    if st.button("Search", key="search_button"):
        all_notes    = get_all_notes_from_db()
        matching_ids = find_relevant_ids_with_openai(query, all_notes)
        if not matching_ids:
            st.info("❌ Brak wyników wyszukiwania.")
        for note in all_notes:
            if note["id"] in matching_ids:
                found = False
                with st.container():
                    user_path = os.path.join(save_dir, f"{note['id']}.png")
                    if os.path.exists(user_path):
                        st.image(user_path, caption=f"Użytkownik {note['id']}", use_container_width=True)
                    else:
                        for ext in (".png", ".jpg", ".jpeg"):
                            stock_path = os.path.join("stock_photo", f"{note['id']}{ext}")
                            if os.path.exists(stock_path):
                                st.image(stock_path, caption=f"Stock {note['id']}", use_container_width=True)
                                found = True
                                break
                    with st.expander("📖 Description"):
                        st.markdown(note["text"])

elif tab == "Reset":
    # ========== RESET ==========
    if st.button("🔄 Restart MyGallery App"):
        for fname in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, fname))

        qdrant_client = get_qdrant_client()
        session_ids = st.session_state["session_ids"]
        if session_ids:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=session_ids,
                wait=True
            )

        st.session_state.clear()

        # Od razu ustaw zakładkę "Gallery" po resecie
        st.session_state["uploader_key"]        = 1
        st.session_state["session_ids"]         = []
        st.session_state["search_query"]        = ""
        st.session_state["search_query_input"]  = ""
        st.success("Data has been reset")
        st.rerun()