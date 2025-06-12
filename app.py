import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from streamlit_searchbox import st_searchbox
import os
import io

st.set_page_config(page_title="BOQ AI Assistant", layout="wide")

# --- Constants ---
CORRECTIONS_FILE = "corrections.csv"

# --- Load Pre-built Index and Data (Fast) ---
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

@st.cache_data
def load_corrections():
    """Loads the user's correction file."""
    if os.path.exists(CORRECTIONS_FILE):
        return pd.read_csv(CORRECTIONS_FILE).set_index('original_description')
    return pd.DataFrame()

def save_new_correction(original_desc, mat_desc, mat_code, lab_desc, lab_code):
    """Saves a single new correction to the CSV file."""
    new_correction = {
        "original_description": [original_desc],
        "mat_description": [mat_desc], "mat_code": [mat_code],
        "lab_description": [lab_desc], "lab_code": [lab_code]
    }
    new_df = pd.DataFrame(new_correction)
    
    if not os.path.exists(CORRECTIONS_FILE):
        new_df.to_csv(CORRECTIONS_FILE, index=False)
    else:
        df = pd.read_csv(CORRECTIONS_FILE)
        df = df[df.original_description != original_desc]
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CORRECTIONS_FILE, index=False)

def search_for_searchbox(query: str, system_type: str, systems: dict, k: int=10) -> list[str]:
    """A wrapper for st_searchbox that returns a list of description strings."""
    if not query or not systems: return []
    model = systems["model"]
    index = systems[system_type]["index"]
    df_db = systems[system_type]["db"]
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return [df_db.iloc[idx]['DB_Description'] for idx in indices[0] if idx != -1]

# --- Main App ---
st.title("Mad-Match (v7.2)")

systems = load_search_systems()
corrections_db = load_corrections()

if systems:
    if 'boq_df' not in st.session_state:
        st.session_state.boq_df = None

    uploaded_file = st.file_uploader("อัปโหลดไฟล์ BOQ.xlsx ของคุณ", type=["xlsx"])

    if uploaded_file:
        if st.session_state.boq_df is None:
            st.session_state.boq_df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.session_state.boq_df['Description'] = st.session_state.boq_df['Description'].astype(str)
        
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
                # --- LEARNING CHECK ---
                # Check if an edit override has been triggered for this row
                edit_mode = st.session_state.get(f'edit_mode_{i}', False)

                if query in corrections_db.index and not edit_mode:
                    learned = corrections_db.loc[query]
                    with row_cols[1]:
                        st.success(f"{learned['mat_description']}", icon="✅")
                        if st.button("แก้ไข", key=f"edit_mat_{i}", type="secondary"):
                            st.session_state[f'edit_mode_{i}'] = True
                            st.rerun() # Rerun to enter edit mode
                    row_cols[2].text_input("mat_code", value=learned['mat_code'], key=f"mat_code_{i}", label_visibility="collapsed", disabled=True)
                    
                    with row_cols[3]:
                        st.success(f"{learned['lab_description']}", icon="✅")
                        if st.button("แก้ไข", key=f"edit_lab_{i}", type="secondary"):
                            st.session_state[f'edit_mode_{i}'] = True
                            st.rerun() # Rerun to enter edit mode
                    row_cols[4].text_input("lab_code", value=learned['lab_code'], key=f"lab_code_{i}", label_visibility="collapsed", disabled=True)
                    
                    st.session_state[f'final_mat_desc_{i}'] = learned['mat_description']
                    st.session_state[f'final_lab_desc_{i}'] = learned['lab_description']
                    st.session_state[f'final_mat_code_{i}'] = learned['mat_code']
                    st.session_state[f'final_lab_code_{i}'] = learned['lab_code']
                
                else: # --- AI SEARCH & MANUAL INPUT MODE ---
                    # --- Material ---
                    final_mat_desc = ""
                    mat_code_val = ""
                    with row_cols[1]:
                        mat_results = search_for_searchbox(query, "material", systems, k=5)
                        mat_options = mat_results + ["โปรดกรอกข้อมูลที่ถูกต้อง", "None"]
                        selected_mat_via_selectbox = st.selectbox("mat_sel", mat_options, index=0 if mat_results else len(mat_options)-2, key=f"mat_select_{i}", label_visibility="collapsed")
                        
                        final_mat_desc = selected_mat_via_selectbox
                        if selected_mat_via_selectbox == "โปรดกรอกข้อมูลที่ถูกต้อง":
                            manual_mat_search = st_searchbox(lambda q: search_for_searchbox(q, "material", systems), key=f"mat_searchbox_{i}", placeholder="พิมพ์เพื่อค้นหาวัสดุ...")
                            if manual_mat_search:
                                final_mat_desc = manual_mat_search

                    if final_mat_desc and final_mat_desc not in ["โปรดกรอกข้อมูลที่ถูกต้อง", "None"]:
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
                        lab_options = lab_results + ["โปรดกรอกข้อมูลที่ถูกต้อง", "None"]
                        selected_lab_via_selectbox = st.selectbox("lab_sel", lab_options, index=0 if lab_results else len(lab_options)-2, key=f"lab_select_{i}", label_visibility="collapsed")

                        final_lab_desc = selected_lab_via_selectbox
                        if selected_lab_via_selectbox == "โปรดกรอกข้อมูลที่ถูกต้อง":
                            manual_lab_search = st_searchbox(lambda q: search_for_searchbox(q, "labour", systems), key=f"lab_searchbox_{i}", placeholder="พิมพ์เพื่อค้นหาค่าแรง...")
                            if manual_lab_search:
                                final_lab_desc = manual_lab_search

                    if final_lab_desc and final_lab_desc not in ["โปรดกรอกข้อมูลที่ถูกต้อง", "None"]:
                        code_row = systems["labour"]["db"].loc[systems["labour"]["db"]['DB_Description'] == final_lab_desc, 'Code']
                        if not code_row.empty: lab_code_val = code_row.iloc[0]
                    row_cols[4].text_input("lab_code", value=lab_code_val, key=f"lab_code_{i}", label_visibility="collapsed")
                    st.session_state[f'final_lab_desc_{i}'] = final_lab_desc
                    st.session_state[f'final_lab_code_{i}'] = lab_code_val
        
        # --- Buttons and Export Section ---
        st.divider()
        st.header("บันทึกและส่งออก (Save & Export)")
        
        export_col, learn_col = st.columns(2)
        with learn_col:
            if st.button("บันทึกการเรียนรู้ (Save Learnings)"):
                with st.spinner("Saving corrections..."):
                    for i, row in boq_df.iterrows():
                        if row['Description'].strip():
                            mat_desc = st.session_state.get(f"final_mat_desc_{i}", "None")
                            lab_desc = st.session_state.get(f"final_lab_desc_{i}", "None")
                            if (mat_desc not in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"]) or (lab_desc not in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"]):
                                save_new_correction(
                                    row['Description'].strip(),
                                    mat_desc if mat_desc not in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"] else "",
                                    st.session_state.get(f"mat_code_{i}", ""),
                                    lab_desc if lab_desc not in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"] else "",
                                    st.session_state.get(f"lab_code_{i}", "")
                                )
                st.success("บันทึกการเรียนรู้สำเร็จ! การเปลี่ยนแปลงจะแสดงผลในครั้งถัดไป")
                st.cache_data.clear()

        with export_col:
            if st.button("เตรียมข้อมูลสำหรับ Export"):
                exported_data = []
                for i, row in boq_df.iterrows():
                    final_mat_desc = st.session_state.get(f'final_mat_desc_{i}', "None")
                    final_lab_desc = st.session_state.get(f'final_lab_desc_{i}', "None")
                    exported_data.append({
                        "Original Description": row['Description'],
                        "Mat Description": "" if final_mat_desc in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"] else final_mat_desc,
                        "Mat Code": st.session_state.get(f"mat_code_{i}", ""),
                        "Lab Description": "" if final_lab_desc in ["None", "โปรดกรอกข้อมูลที่ถูกต้อง"] else final_lab_desc,
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