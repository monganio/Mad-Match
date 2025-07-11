import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import io
from streamlit_searchbox import st_searchbox
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# Firestore Init
# -------------------------
@st.cache_resource
def init_firestore():
    try:
        cred = credentials.Certificate(st.secrets["firebase_service_account"])
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"เชื่อมต่อ Firebase ไม่สำเร็จ: {e}")
        return None

db = init_firestore()

# -------------------------
# FAISS, Model, DB Load
# -------------------------
@st.cache_resource
def load_search_systems():
    systems = {}
    try:
        systems["material"] = {
            "index": faiss.read_index("material.index"),
            "db": pd.read_pickle("material_db.pkl")
        }
        systems["labour"] = {
            "index": faiss.read_index("labour.index"),
            "db": pd.read_pickle("labour_db.pkl")
        }
        systems["model"] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return systems
    except Exception as e:
        st.error(f"Error loading search systems: {e}")
        return None

# -------------------------
# Load & Save to Firestore
# -------------------------
def load_corrections_from_firestore():
    if not db:
        return pd.DataFrame()
    docs = db.collection("boq_corrections").stream()
    data = []
    for doc in docs:
        item = doc.to_dict()
        item["original_description"] = doc.id
        data.append(item)
    df = pd.DataFrame(data)
    return df.set_index("original_description") if not df.empty else pd.DataFrame()

def save_correction_to_firestore(description, mat_desc, mat_code, lab_desc, lab_code, is_verified, edited_by):
    if not db:
        return
    doc_ref = db.collection("boq_corrections").document(description)
    doc_ref.set({
        "mat_description": mat_desc,
        "mat_code": mat_code,
        "lab_description": lab_desc,
        "lab_code": lab_code,
        "is_verified": is_verified,
        "edited_by": edited_by,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

# -------------------------
# Search & UI
# -------------------------
def search(query: str, system_type: str, systems: dict, k: int = 5):
    if not query or not systems:
        return []
    model = systems["model"]
    index = systems[system_type]["index"]
    df_db = systems[system_type]["db"]
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, k)
    return [df_db.iloc[idx]['DB_Description'] for idx in indices[0] if idx != -1]

def get_code_by_description(df, description):
    code_row = df.loc[df['DB_Description'] == description, 'Code']
    return code_row.iloc[0] if not code_row.empty else ""

def render_selectbox(col, query, system_key, systems, select_key, searchbox_key, placeholder):
    options = search(query, system_key, systems) + ["กรอกข้อมูลเอง", "None"]
    selected = col.selectbox("", options, key=select_key, label_visibility="collapsed")
    final = selected
    if selected == "กรอกข้อมูลเอง":
        manual = st_searchbox(lambda q: search(q, system_key, systems), key=searchbox_key, placeholder=placeholder)
        if manual:
            final = manual
    return final

# -------------------------
# Main App
# -------------------------
st.set_page_config(page_title="BOQ Assistant (Firebase)", layout="wide")
st.title("Mad-Match (Firebase Edition)")

systems = load_search_systems()
corrections_db = load_corrections_from_firestore()

if 'boq_df' not in st.session_state:
    st.session_state.boq_df = None

st.session_state.user_name = st.text_input("ชื่อของคุณ (สำหรับบันทึก)", value=st.session_state.get("user_name", ""))
if not st.session_state.user_name:
    st.warning("กรุณากรอกชื่อก่อนเริ่มใช้งาน")
    st.stop()

uploaded_file = st.file_uploader("อัปโหลดไฟล์ BOQ.xlsx ของคุณ", type=["xlsx"])
if uploaded_file:
    if st.session_state.boq_df is None:
        try:
            st.session_state.boq_df = pd.read_excel(uploaded_file, engine='openpyxl')
            if 'Description' not in st.session_state.boq_df.columns:
                raise KeyError("ไม่พบคอลัมน์ Description")
            st.session_state.boq_df['Description'] = st.session_state.boq_df['Description'].astype(str)
        except Exception as e:
            st.error(f"โหลดไฟล์ไม่สำเร็จ: {e}")
            st.stop()

    boq_df = st.session_state.boq_df
    st.header("ตรวจสอบและแก้ไขข้อมูล")

    header_cols = st.columns([4, 3, 2, 3, 2, 2])
    headers = ["Original Description", "Material", "Mat Code", "Labour", "Lab Code", "✓"]
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    st.divider()

    for i, row in boq_df.iterrows():
        query = row['Description'].strip()
        row_cols = st.columns([4, 3, 2, 3, 2, 2])
        row_cols[0].text(query)

        learned = corrections_db.loc[query] if query in corrections_db.index else {}

        mat_desc = render_selectbox(row_cols[1], query, "material", systems, f"mat_sel_{i}", f"mat_search_{i}", "พิมพ์วัสดุ...")
        mat_code = get_code_by_description(systems["material"]["db"], mat_desc)
        row_cols[2].text_input("mat_code", value=mat_code, key=f"mat_code_{i}", label_visibility="collapsed")

        lab_desc = render_selectbox(row_cols[3], query, "labour", systems, f"lab_sel_{i}", f"lab_search_{i}", "พิมพ์แรงงาน...")
        lab_code = get_code_by_description(systems["labour"]["db"], lab_desc)
        row_cols[4].text_input("lab_code", value=lab_code, key=f"lab_code_{i}", label_visibility="collapsed")

        verified_key = f"verified_{i}"
        verified = learned.get("is_verified", False)
        st.session_state[verified_key] = row_cols[5].checkbox("", value=verified, key=verified_key)

        st.session_state[f"final_{i}"] = {
            "query": query,
            "mat_desc": mat_desc,
            "mat_code": mat_code,
            "lab_desc": lab_desc,
            "lab_code": lab_code,
            "verified": st.session_state[verified_key]
        }

    st.divider()
    st.header("บันทึกและ Export")

    if st.button("บันทึกทั้งหมดไป Firestore"):
        with st.spinner("กำลังบันทึก..."):
            for i in range(len(boq_df)):
                final = st.session_state.get(f"final_{i}", {})
                if final:
                    save_correction_to_firestore(
                        final["query"],
                        final["mat_desc"], final["mat_code"],
                        final["lab_desc"], final["lab_code"],
                        final["verified"], st.session_state.user_name
                    )
        st.success("บันทึกเรียบร้อย")

    if st.button("เตรียม Export เป็น Excel"):
        export_data = []
        for i in range(len(boq_df)):
            final = st.session_state.get(f"final_{i}", {})
            export_data.append({
                "Original Description": final.get("query", ""),
                "Mat Description": final.get("mat_desc", ""),
                "Mat Code": final.get("mat_code", ""),
                "Lab Description": final.get("lab_desc", ""),
                "Lab Code": final.get("lab_code", ""),
                "Verified": final.get("verified", False)
            })
        st.session_state.export_df = pd.DataFrame(export_data)

    if "export_df" in st.session_state:
        st.dataframe(st.session_state.export_df, use_container_width=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.export_df.to_excel(writer, index=False)
        st.download_button("ดาวน์โหลด Excel", data=output.getvalue(), file_name="BOQ_Exported.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")