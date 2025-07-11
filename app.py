import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from streamlit_searchbox import st_searchbox
import os
import io
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="BOQ AI Assistant", layout="wide")

# -------------------------------
# GOOGLE SHEETS AUTH
# -------------------------------
@st.cache_resource
def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets Authentication Error: {e}")
        return None

SHEET_NAME = "BOQ_Learnings_DB"

# -------------------------------
# LOAD FAISS, MODEL, DB
# -------------------------------
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
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Run `create_index.py` first.")
        return None

@st.cache_data(ttl=300)
def load_corrections_from_gsheet(sheet_name):
    client = get_gsheet_client()
    if not client or not sheet_name:
        return pd.DataFrame()
    try:
        sheet = client.open(sheet_name).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        return df.set_index('original_description') if 'original_description' in df.columns else pd.DataFrame()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Sheet '{sheet_name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load from Google Sheet: {e}")
        return pd.DataFrame()

def save_correction_to_gsheet(client, sheet_name, original_desc, mat_desc, mat_code, lab_desc, lab_code):
    if not client or not sheet_name:
        return
    try:
        sheet = client.open(sheet_name).sheet1
        try:
            cell = sheet.find(original_desc, in_column=1)
        except gspread.exceptions.CellNotFound:
            cell = None
        new_row = [original_desc, mat_desc, mat_code, lab_desc, lab_code]
        if cell:
            sheet.update(f'A{cell.row}:E{cell.row}', [new_row])
        else:
            sheet.append_row(new_row, value_input_option='USER_ENTERED')
    except Exception as e:
        st.warning(f"Failed to save learning: {e}")

# -------------------------------
# SEARCH FUNCTIONS
# -------------------------------
def search_for_searchbox(query: str, system_type: str, systems: dict, k: int=10) -> list[str]:
    if not query or not systems:
        return []
    model = systems["model"]
    index = systems[system_type]["index"]
    df_db = systems[system_type]["db"]
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return [df_db.iloc[idx]['DB_Description'] for idx in indices[0] if idx != -1]

# -------------------------------
# HELPER UI FUNCTIONS
# -------------------------------
def get_code_by_description(df, description):
    code_row = df.loc[df['DB_Description'] == description, 'Code']
    return code_row.iloc[0] if not code_row.empty else ""

def render_selectbox_with_manual_input(col, label_key, query, system_type, systems, placeholder, select_key, manual_key):
    options = search_for_searchbox(query, system_type, systems, k=5) + ["กรอกข้อมูลเอง", "None"]
    selected = col.selectbox(label_key, options, index=0 if options[0] != "None" else len(options)-1, key=select_key, label_visibility="collapsed")
    final_text = selected
    if selected == "กรอกข้อมูลเอง":
        manual_result = st_searchbox(lambda q: search_for_searchbox(q, system_type, systems), key=manual_key, placeholder=placeholder)
        if manual_result:
            final_text = manual_result
    return final_text

def handle_edit_mode(i, row, corrections_db):
    query = row['Description'].strip()
    if query in corrections_db.index and not st.session_state.get(f'edit_mode_{i}', False):
        learned = corrections_db.loc[query]
        return {
            "editing": False,
            "mat_desc": learned['mat_description'],
            "mat_code": learned['mat_code'],
            "lab_desc": learned['lab_description'],
            "lab_code": learned['lab_code']
        }
    return { "editing": True }

def render_result_row(i, row, systems, corrections_db):
    query = row['Description'].strip()
    row_cols = st.columns([4, 3, 2, 3, 2])
    row_cols[0].text(query if query else "---")

    if not query:
        return

    state = handle_edit_mode(i, row, corrections_db)

    if not state["editing"]:
        with row_cols[1]:
            st.success(state['mat_desc'], icon="✅")
            if st.button("แก้ไข", key=f"edit_mat_{i}", type="secondary"):
                st.session_state[f'edit_mode_{i}'] = True
                st.rerun()
        row_cols[2].text_input("mat_code", value=state['mat_code'], key=f"mat_code_{i}", label_visibility="collapsed", disabled=True)
        with row_cols[3]:
            st.success(state['lab_desc'], icon="✅")
            if st.button("แก้ไข", key=f"edit_lab_{i}", type="secondary"):
                st.session_state[f'edit_mode_{i}'] = True
                st.rerun()
        row_cols[4].text_input("lab_code", value=state['lab_code'], key=f"lab_code_{i}", label_visibility="collapsed", disabled=True)

        st.session_state[f'final_mat_desc_{i}'] = state['mat_desc']
        st.session_state[f'final_mat_code_{i}'] = state['mat_code']
        st.session_state[f'final_lab_desc_{i}'] = state['lab_desc']
        st.session_state[f'final_lab_code_{i}'] = state['lab_code']
    else:
        mat_desc = render_selectbox_with_manual_input(
            row_cols[1], "mat_sel", query, "material", systems,
            "พิมพ์เพื่อค้นหาวัสดุ...", f"mat_select_{i}", f"mat_searchbox_{i}"
        )
        mat_code = get_code_by_description(systems["material"]["db"], mat_desc)
        row_cols[2].text_input("mat_code", value=mat_code, key=f"mat_code_{i}", label_visibility="collapsed")

        lab_desc = render_selectbox_with_manual_input(
            row_cols[3], "lab_sel", query, "labour", systems,
            "พิมพ์เพื่อค้นหา...", f"lab_select_{i}", f"lab_searchbox_{i}"
        )
        lab_code = get_code_by_description(systems["labour"]["db"], lab_desc)
        row_cols[4].text_input("lab_code", value=lab_code, key=f"lab_code_{i}", label_visibility="collapsed")

        st.session_state[f'final_mat_desc_{i}'] = mat_desc
        st.session_state[f'final_mat_code_{i}'] = mat_code
        st.session_state[f'final_lab_desc_{i}'] = lab_desc
        st.session_state[f'final_lab_code_{i}'] = lab_code

# -------------------------------
# MAIN APP
# -------------------------------
st.title("Mad-Match (v8.0)")

client = get_gsheet_client()
systems = load_search_systems()

if systems and client:
    corrections_db = load_corrections_from_gsheet(SHEET_NAME)

    if 'boq_df' not in st.session_state:
        st.session_state.boq_df = None

    uploaded_file = st.file_uploader("อัปโหลดไฟล์ BOQ.xlsx ของคุณ", type=["xlsx"])

    if uploaded_file:
        if st.session_state.boq_df is None:
            try:
                st.session_state.boq_df = pd.read_excel(uploaded_file, engine='openpyxl')
                if 'Description' not in st.session_state.boq_df.columns:
                    raise KeyError("ไม่พบคอลัมน์ Description")
                st.session_state.boq_df['Description'] = st.session_state.boq_df['Description'].astype(str)
            except Exception as e:
                st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
                if st.button("อัปโหลดไฟล์ใหม่"):
                    st.session_state.boq_df = None
                    st.experimental_rerun()
                st.stop()

        boq_df = st.session_state.boq_df
        st.header("ตรวจสอบและแก้ไขข้อมูล")

        header_cols = st.columns([4, 3, 2, 3, 2])
        header_cols[0].markdown("**Original Description**")
        header_cols[1].markdown("**AI / Manual Search (Material)**")
        header_cols[2].markdown("**Mat. Code**")
        header_cols[3].markdown("**AI / Manual Search (Labour/Sup)**")
        header_cols[4].markdown("**Lab./Sup. Code**")
        st.divider()

        for i, row in boq_df.iterrows():
            render_result_row(i, row, systems, corrections_db)

        st.divider()
        st.header("บันทึกและส่งออก (Save & Export)")

        export_col, learn_col = st.columns(2)
        with learn_col:
            if st.button("บันทึกการเรียนรู้ของ MadMatch"):
                with st.spinner("กำลังบันทึก..."):
                    for i, row in boq_df.iterrows():
                        if row['Description'].strip():
                            mat_desc = st.session_state.get(f"final_mat_desc_{i}", "None")
                            lab_desc = st.session_state.get(f"final_lab_desc_{i}", "None")
                            if mat_desc not in ["None", "กรอกข้อมูลเอง"] or lab_desc not in ["None", "กรอกข้อมูลเอง"]:
                                save_correction_to_gsheet(
                                    client, SHEET_NAME,
                                    row['Description'].strip(),
                                    mat_desc if mat_desc not in ["None", "กรอกข้อมูลเอง"] else "",
                                    st.session_state.get(f"mat_code_{i}", ""),
                                    lab_desc if lab_desc not in ["None", "กรอกข้อมูลเอง"] else "",
                                    st.session_state.get(f"lab_code_{i}", "")
                                )
                st.success("บันทึกสำเร็จ")
                st.cache_data.clear()

        with export_col:
            if st.button("เตรียมข้อมูลสำหรับ Export"):
                exported_data = []
                for i, row in boq_df.iterrows():
                    exported_data.append({
                        "Original Description": row['Description'],
                        "Mat Description": st.session_state.get(f'final_mat_desc_{i}', ""),
                        "Mat Code": st.session_state.get(f"mat_code_{i}", ""),
                        "Lab Description": st.session_state.get(f'final_lab_desc_{i}', ""),
                        "Lab Code": st.session_state.get(f"lab_code_{i}", "")
                    })
                st.session_state.export_df = pd.DataFrame(exported_data)

        if 'export_df' in st.session_state:
            st.dataframe(st.session_state.export_df, use_container_width=True)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.export_df.to_excel(writer, index=False, sheet_name='BOQ_Processed')
            excel_data = output.getvalue()
            st.download_button("ดาวน์โหลดไฟล์ Excel", data=excel_data, file_name="BOQ_Processed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")