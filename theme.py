from matplotlib import style
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

        /* ===== MAIN BACKGROUND ===== */
        .stApp {{
            background: linear-gradient(rgba(10,31,68,0.90), rgba(10,31,68,0.90)),
                        url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
            color: #FFFFFF !important;
        }}

        /* ===== GLOBAL TEXT FIX ===== */
        html, body, [class*="css"] {{
            color: #FFFFFF !important;
        }}

        /* ===== HEADINGS ===== */
        h1, h2, h3, h4 {{
            color: #E6C200 !important;
            font-weight: 700;
        }}

        /* ===== TABS ===== */
        .stTabs [role="tab"] {{
            color: #FFFFFF !important;
            font-weight: 600;
            padding: 10px 18px;
        }}

        .stTabs [aria-selected="true"] {{
            color: #E6C200 !important;
            border-bottom: 3px solid #E6C200 !important;
        }}

        /* ===== BUTTONS ===== */
        .stButton>button {{
            background: linear-gradient(135deg, #1f3b73, #162b50);
            color: #FFFFFF;
            border-radius: 10px;
            border: 1px solid #E6C200;
            font-weight: 600;
            padding: 8px 18px;
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg, #2d4ea3, #1f3b73);
            color: #E6C200;
            border: 1px solid #E6C200;
        }}

        /* ===== SUCCESS ALERT ===== */
        div[data-testid="stAlert"] {{
            background: rgba(0,245,255,0.12) !important;
            border: 1px solid #00F5FF !important;
            color: #FFFFFF !important;
            border-radius: 10px;
        }}

        div[data-testid="stAlert"] p {{
            color: #FFFFFF !important;
        }}

        /* ===== METRICS ===== */
        div[data-testid="metric-container"] {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(230,194,0,0.4);
            padding: 18px;
            border-radius: 14px;
        }}

        div[data-testid="metric-container"] label {{
            color: #E6C200 !important;
            font-size: 15px !important;
            font-weight: 600;
        }}

        div[data-testid="metric-container"] div {{
            color: #FFFFFF !important;
            font-size: 22px;
            font-weight: bold !important;
        }}

        /* ===== INPUT LABELS ===== */
        label {{
            color: #E6C200 !important;
            font-weight: 600 !important;
        }}

        /* ===== INPUT FIELDS ===== */
        input, textarea {{
            background-color: rgba(255,255,255,0.08) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(230,194,0,0.4) !important;
            border-radius: 8px !important;
        }}

        /* ===== SELECTBOX & MULTISELECT FIX ===== */
        /* ============================= */
        /* SELECTBOX / MULTISELECT FIX  */
        /* ============================= */

        /* Closed select container */
        div[data-baseweb="select"] > div {{
            background: transparent !important;
            border: 1px solid rgba(230,194,0,0.6) !important;
            border-radius: 8px !important;
        }}

        /* ACTUAL selected value text (real fix) */
        div[data-baseweb="select"] div[class*="singleValue"] {{
            color: #FFFFFF !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }}

        /* Fallback for newer builds */
        div[data-baseweb="select"] div {{
            color: #FFFFFF !important;
        }}

        /* Placeholder */
        div[data-baseweb="select"] input {{
            color: #FFFFFF !important;
        }}

        /* Dropdown menu */
        div[role="listbox"] {{
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
        }}

        /* Dropdown options */
        div[role="option"] {{
            color: #0A1F44 !important;
            font-weight: 500 !important;
        }}

        /* Hover */
        div[role="option"]:hover {{
            background-color: #E6C200 !important;
            color: #0A1F44 !important;
        }}
        /* ===== RADIO BUTTON FIX (THIS WAS THE ISSUE) ===== */

        div[role="radiogroup"] * {{
            color: #FFFFFF !important;
        }}

        div[role="radiogroup"] label {{
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }}

        div[role="radiogroup"] span {{
            color: #FFFFFF !important;
        }}

        /* ===== DATAFRAME ===== */
        .stDataFrame {{
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 12px;
            color: #FFFFFF !important;
        }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {{
            background-color: rgba(10,31,68,0.95);
        }}

        section[data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}

        section[data-testid="stSidebar"] a:hover {{
            color: #E6C200 !important;
        }}

        /* ===== SCROLLBAR ===== */
        div[data-testid="stAppViewContainer"]::-webkit-scrollbar {{
            width: 12px;
        }}

        div[data-testid="stAppViewContainer"]::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.05);
        }}

        div[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, #E6C200, #b38f00);
            border-radius: 8px;
        }}

        div[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(180deg, #FFD700, #E6C200);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )