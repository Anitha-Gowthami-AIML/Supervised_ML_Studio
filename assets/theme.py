import streamlit as st
import base64
import os


def apply_theme(background_image):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, background_image)

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* =========================
           MAIN APP BACKGROUND
        ========================== */
        .stApp {{
            background: linear-gradient(rgba(10,31,68,0.92), rgba(10,31,68,0.92)),
                        url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* =========================
           HEADINGS - BRIGHT YELLOW
        ========================== */
        h1, h2, h3, h4, h5 {{
            color: #FFD700 !important;
            font-weight: 700;
        }}

        /* =========================
           NORMAL TEXT ON DARK BG
        ========================== */
        .stMarkdown, p, span, label {{
            color: #FFFFFF;
        }}

        /* =========================
           SIDEBAR
        ========================== */
        section[data-testid="stSidebar"] {{
            background-color: rgba(10,31,68,0.95);
        }}

        section[data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}

        /* =========================
           BUTTONS
        ========================== */
        .stButton>button {{
            background: linear-gradient(135deg, #1f3b73, #162b50);
            color: #FFD700;
            border: 1px solid #FFD700;
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton>button:hover {{
            background: #FFD700;
            color: #0A1F44;
        }}

        /* =========================
           INPUT FIELDS (WHITE BOX → BLACK TEXT)
        ========================== */
        input, textarea {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-radius: 8px;
        }}

        /* =========================
           SELECTBOX CLOSED STATE
        ========================== */
        div[data-baseweb="select"] > div {{
            background: #FFFFFF !important;
            color: #000000 !important;
        }}

        div[data-baseweb="select"] div[class*="singleValue"] {{
            color: #000000 !important;
        }}

        /* =========================
           DROPDOWN MENU
        ========================== */
        div[role="listbox"] {{
            background-color: #FFFFFF !important;
        }}

        div[role="option"] {{
            color: #000000 !important;
        }}

        div[role="option"]:hover {{
            background-color: #FFD700 !important;
            color: #000000 !important;
        }}

        /* =========================
           RADIO BUTTONS
        ========================== */
        div[role="radiogroup"] label {{
            color: #FFFFFF !important;
        }}

        /* =========================
           DATAFRAME
        ========================== */
        .stDataFrame {{
            background-color: rgba(255,255,255,0.05);
            color: #FFFFFF !important;
        }}

        /* =========================
           METRICS
        ========================== */
        div[data-testid="metric-container"] {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px;
        }}

        div[data-testid="metric-container"] label {{
            color: #FFD700 !important;
        }}

        div[data-testid="metric-container"] div {{
            color: #FFFFFF !important;
        }}

        /* =========================
           ALERTS
        ========================== */
        div[data-testid="stAlert"] {{
            background-color: rgba(255,255,255,0.1) !important;
            color: #FFFFFF !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )