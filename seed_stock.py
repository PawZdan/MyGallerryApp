import os, re, base64
from io import BytesIO
from openai import OpenAI
from dotenv import dotenv_values
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter

# --- Konfiguracja ---
env = dotenv_values(".env")
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
qdrant = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"])
COL_NAME   = "images_embs"
STOCK_DIR  = r"C:\Users\Paweł\Desktop\od zera do chuja\modul 8\mod_8_zad_1_image_finder\stock_photo"
EMBED_DIM  = 3072

# --- Upewnij się, że kolekcja istnieje i jest czysta dla stockowych zdjęć ---
if not qdrant.collection_exists(COL_NAME):
    qdrant.create_collection(
        collection_name=COL_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

# Usuń wszystkie dotychczasowe stockowe punkty
qdrant.delete(
    collection_name=COL_NAME,
    points_selector=Filter(must=[{"key":"stock","match":{"value":True}}]),
    wait=True
)

# --- Pomocnicze funkcje ---
def describe_image(path: str) -> str:
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = [
        {"type":"text","text":"Stwórz opis obrazka opisz co na nim sie znajduje"},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}","detail":"high"}}
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

def embed_text(text: str) -> list[float]:
    r = openai_client.embeddings.create(input=[text], model="text-embedding-3-large")
    return r.data[0].embedding

def extract_number(fname: str) -> int:
    # Dopasuj liczbę na początku nazwy pliku
    m = re.match(r"^(\d+)", fname)
    return int(m.group(1)) if m else 0

# --- Wczytanie i sortowanie plików ---
files = [
    f for f in os.listdir(STOCK_DIR)
    if re.match(r"^\d+\.(png|jpg|jpeg)$", f, re.IGNORECASE)
]
# Sortuj po wyekstrahowanym numerze
files.sort(key=extract_number)

# --- Seedowanie kolekcji ---
for idx, fname in enumerate(files, start=1):
    path = os.path.join(STOCK_DIR, fname)
    text = describe_image(path)
    emb  = embed_text(text)
    qdrant.upsert(
        collection_name=COL_NAME,
        points=[PointStruct(
            id=idx,
            vector=emb,
            payload={"text": text, "stock": True, "filename": fname}
        )]
    )
    print(f"✔️ Wgrano {fname} jako ID={idx}")