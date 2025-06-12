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

# --- Google Sheets Authentication ---
@st.cache_resource
def get_gsheet_client():
    """Authenticates with Google Sheets using Streamlit Secrets."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Google Sheets Authentication Error: {e}. Please check your Streamlit Secrets setup.")
        return None

# Use st.text_input to get the sheet name from the user for more flexibility
SHEET_NAME = "BOQ_Learnings_DB"

# --- Load Systems (FAISS, Model) ---
@st.cache_resource
def load_search_systems():
    """Loads FAISS indexes, databases, and the model."""
    systems = {}
    try:
        systems["material"] = {"index": faiss.read_index("material.index"), "db": pd.read_pickle("material_db.pkl")}
        systems["labour"] = {"index": faiss.read_index("labour.index"), "db": pd.read_pickle("labour_db.pkl")}
        systems["model"] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return systems
    except FileNotFoundError as e:
        st.error(f"Error: Could not find index file '{e.filename}'. Please run `create_index.py` first.")
        return None

@st.cache_data(ttl=300) # Cache for 5 minutes
def load_corrections_from_gsheet(sheet_name):  # <--- เอา client ออก
    """Loads the user's corrections from the specified Google Sheet."""
    client = get_gsheet_client() # <--- เรียกใช้ฟังก์ชันเพื่อดึง client ที่แคชไว้
    if not client or not sheet_name: return pd.DataFrame()
    try:
        sheet = client.open(sheet_name).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        if 'original_description' in df.columns:
            return df.set_index('original_description')
        return pd.DataFrame()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Google Sheet '{sheet_name}' not found. Please check the name or share permissions.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load from Google Sheet: {e}")
        return pd.DataFrame()

def save_correction_to_gsheet(client, sheet_name, original_desc, mat_desc, mat_code, lab_desc, lab_code):
    """Saves a single new correction to the Google Sheet, overwriting if exists."""
    if not client or not sheet_name: return
    try:
        sheet = client.open(sheet_name).sheet1
        # Find if the row exists
        cell = None
        try:
            cell = sheet.find(original_desc, in_column=1)
        except gspread.exceptions.CellNotFound:
            pass # It's a new entry
        
        new_row = [original_desc, mat_desc, mat_code, lab_desc, lab_code]
        if cell:
            # Update existing row
            sheet.update(f'A{cell.row}:E{cell.row}', [new_row])
        else:
            # Append new row
            sheet.append_row(new_row, value_input_option='USER_ENTERED')
    except Exception as e:
        st.warning(f"Could not save learning to Google Sheet: {e}")

def search_for_searchbox(query: str, system_type: str, systems: dict, k: int=10) -> list[str]:
    # ... (same as before) ...
    if not query or not systems: return []
    model = systems["model"]
    index = systems[system_type]["index"]
    df_db = systems[system_type]["db"]
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return [df_db.iloc[idx]['DB_Description'] for idx in indices[0] if idx != -1]

# --- Main App ---
st.title("Mad-Match (v8.0)")

client = get_gsheet_client()
systems = load_search_systems()

if systems and client:
    corrections_db = load_corrections_from_gsheet(SHEET_NAME)

    if 'boq_df' not in st.session_state:
        st.session_state.boq_df = None

    uploaded_file = st.file_uploader("อัปโหลดไฟล์ BOQ.xlsx ของคุณ", type=["xlsx"])

    if uploaded_file:
        # ... (The entire UI loop and logic from v7.2 remains here, no changes needed in this part) ...
        # For brevity, I'll put a placeholder here. Use the code from v7.2 for the UI.
        # --- Start of UI code from v7.2 ---
        if st.session_state.boq_df is None:
            try:
                st.session_state.boq_df = pd.read_excel(uploaded_file, engine='openpyxl')
                if 'Description' not in st.session_state.boq_df.columns:
                    raise KeyError("ไม่พบคอลัมน์ Description ในไฟล์ที่อัปโหลด")
                st.session_state.boq_df['Description'] = st.session_state.boq_df['Description'].astype(str)
            except KeyError as ke:
                st.error(f"ข้อผิดพลาด: {ke}")
                if st.button("อัปโหลดไฟล์ใหม่"):
                    st.session_state.boq_df = None
                    st.experimental_rerun()
                st.stop()
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
            query = row['Description'].strip()
            row_cols = st.columns([4, 3, 2, 3, 2])
            row_cols[0].text(query if query else "---")

            if query:
                edit_mode = st.session_state.get(f'edit_mode_{i}', False)

                if query in corrections_db.index and not edit_mode:
                    learned = corrections_db.loc[query]
                    with row_cols[1]:
                        st.success(f"{learned['mat_description']}", icon="✅")
                        if st.button("แก้ไข", key=f"edit_mat_{i}", type="secondary"):
                            st.session_state[f'edit_mode_{i}'] = True
                            st.rerun()
                    row_cols[2].text_input("mat_code", value=learned['mat_code'], key=f"mat_code_{i}", label_visibility="collapsed", disabled=True)
                    
                    with row_cols[3]:
                        st.success(f"{learned['lab_description']}", icon="✅")
                        if st.button("แก้ไข", key=f"edit_lab_{i}", type="secondary"):
                            st.session_state[f'edit_mode_{i}'] = True
                            st.rerun()
                    row_cols[4].text_input("lab_code", value=learned['lab_code'], key=f"lab_code_{i}", label_visibility="collapsed", disabled=True)
                    
                    st.session_state[f'final_mat_desc_{i}'] = learned['mat_description']
                    st.session_state[f'final_lab_desc_{i}'] = learned['lab_description']
                    st.session_state[f'final_mat_code_{i}'] = learned['mat_code']
                    st.session_state[f'final_lab_code_{i}'] = learned['lab_code']
                
                else: 
                    # --- Material ---
                    final_mat_desc = ""
                    mat_code_val = ""
                    with row_cols[1]:
                        mat_results = search_for_searchbox(query, "material", systems, k=5)
                        mat_options = mat_results + ["กรอกข้อมูลเอง", "None"]
                        selected_mat_via_selectbox = st.selectbox("mat_sel", mat_options, index=0 if mat_results else len(mat_options)-2, key=f"mat_select_{i}", label_visibility="collapsed")
                        final_mat_desc = selected_mat_via_selectbox
                        if selected_mat_via_selectbox == "กรอกข้อมูลเอง":
                            manual_mat_search = st_searchbox(lambda q: search_for_searchbox(q, "material", systems), key=f"mat_searchbox_{i}", placeholder="พิมพ์เพื่อค้นหาวัสดุ...")
                            if manual_mat_search: final_mat_desc = manual_mat_search
                    if final_mat_desc and final_mat_desc not in ["กรอกข้อมูลเอง", "None"]:
                        code_row = systems["material"]["db"].loc[systems["material"]["db"]['DB_Description'] == final_mat_desc, 'Code']
                        if not code_row.empty: mat_code_val = code_row.iloc[0]
                    row_cols[2].text_input("mat_code", value=mat_code_val, key=f"mat_code_{i}", label_visibility="collapsed")
                    st.session_state[f'final_mat_desc_{i}'] = final_mat_desc
                    st.session_state[f'final_mat_code_{i}'] = mat_code_val

                    # --- Labour ---
                    final_lab_desc = ""
                    lab_code_val = ""
                    with row_cols[3]:
                        lab_results = search_for_searchbox(query, "labour", systems, k=5)
                        lab_options = lab_results + ["กรอกข้อมูลเอง", "None"]
                        selected_lab_via_selectbox = st.selectbox("lab_sel", lab_options, index=0 if lab_results else len(lab_options)-2, key=f"lab_select_{i}", label_visibility="collapsed")
                        final_lab_desc = selected_lab_via_selectbox
                        if selected_lab_via_selectbox == "กรอกข้อมูลเอง":
                            manual_lab_search = st_searchbox(lambda q: search_for_searchbox(q, "labour", systems), key=f"lab_searchbox_{i}", placeholder="พิมพ์เพื่อค้นหา...")
                            if manual_lab_search: final_lab_desc = manual_lab_search
                    if final_lab_desc and final_lab_desc not in ["กรอกข้อมูลเอง", "None"]:
                        code_row = systems["labour"]["db"].loc[systems["labour"]["db"]['DB_Description'] == final_lab_desc, 'Code']
                        if not code_row.empty: lab_code_val = code_row.iloc[0]
                    row_cols[4].text_input("lab_code", value=lab_code_val, key=f"lab_code_{i}", label_visibility="collapsed")
                    st.session_state[f'final_lab_desc_{i}'] = final_lab_desc
                    st.session_state[f'final_lab_code_{i}'] = lab_code_val
        # --- End of UI code from v7.2 ---

        # --- Buttons and Export Section (Logic now points to Google Sheets) ---
        st.divider()
        st.header("บันทึกและส่งออก (Save & Export)")
        
        export_col, learn_col = st.columns(2)
        with learn_col:
            if st.button("บันทึกการเรียนรู้ของ MadMatch"):
                with st.spinner("Saving corrections to Google Sheet..."):
                    for i, row in boq_df.iterrows():
                        if row['Description'].strip():
                            mat_desc = st.session_state.get(f"final_mat_desc_{i}", "None")
                            lab_desc = st.session_state.get(f"final_lab_desc_{i}", "None")
                            if (mat_desc not in ["None", "กรอกข้อมูลเอง"]) or (lab_desc not in ["None", "กรอกข้อมูลเอง"]):
                                save_correction_to_gsheet(
                                    client, SHEET_NAME,
                                    row['Description'].strip(),
                                    mat_desc if mat_desc not in ["None", "กรอกข้อมูลเอง"] else "",
                                    st.session_state.get(f"mat_code_{i}", ""),
                                    lab_desc if lab_desc not in ["None", "กรอกข้อมูลเอง"] else "",
                                    st.session_state.get(f"lab_code_{i}", "")
                                )
                st.success("MadMatch ฉลาดขึ้นอีกขั้น ขอบคุณนะ")
                st.cache_data.clear() # Clear cache to reload corrections on next run

        with export_col:
            if st.button("เตรียมข้อมูลสำหรับ Export"):
                exported_data = []
                for i, row in boq_df.iterrows():
                    final_mat_desc = st.session_state.get(f'final_mat_desc_{i}', "None")
                    final_lab_desc = st.session_state.get(f'final_lab_desc_{i}', "None")
                    exported_data.append({
                        "Original Description": row['Description'],
                        "Mat Description": "" if final_mat_desc in ["None", "กรอกข้อมูลเอง"] else final_mat_desc,
                        "Mat Code": st.session_state.get(f"mat_code_{i}", ""),
                        "Lab Description": "" if final_lab_desc in ["None", "กรอกข้อมูลเอง"] else final_lab_desc,
                        "Lab Code": st.session_state.get(f"lab_code_{i}", "")
                    })
                st.session_state.export_df = pd.DataFrame(exported_data)
        
        if 'export_df' in st.session_state and st.session_state.export_df is not None:
            st.dataframe(st.session_state.export_df, use_container_width=True)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.export_df.to_excel(writer, index=False, sheet_name='BOQ_Processed')
            excel_data = output.getvalue()
            st.download_button("ดาวน์โหลดไฟล์ Excel", data=excel_data, file_name=f"BOQ_Processed_{uploaded_file.name}", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")