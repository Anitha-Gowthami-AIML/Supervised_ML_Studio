import streamlit as st
import base64

st.set_page_config(
    page_title="Supervised ML Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Encode Image ----------
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("ML_image.jpeg")

# ---------- Background + Global CSS ----------
st.markdown(f"""
<style>
html, body, [class*="css"]  {{
    overflow-x: hidden !important;
}}

.stApp {{
    background: linear-gradient(rgba(10,31,68,0.85), rgba(10,31,68,0.85)),
                url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}}

.flow-container {{
    display:flex;
    justify-content:center;
    align-items:center;
    gap:15px;
    flex-wrap:nowrap;
    overflow-x:auto;
    padding-bottom:10px;
}}

.flow-box {{
    min-width:160px;
    height:90px;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:15px;
    border-radius:16px;
    font-size:15px;
    font-weight:600;
    text-align:center;
    color:#0A1F44;
    box-shadow:0px 6px 18px rgba(0,0,0,0.4);
    transition: transform 0.3s ease;
}}

.flow-box:hover {{
    transform: translateY(-6px);
}}

.arrow {{
    font-size:28px;
    font-weight:bold;
    color:#00F5FF;
    text-shadow: 0px 0px 12px rgba(0,245,255,0.8);
}}
</style>
""", unsafe_allow_html=True)

# ---------- Soft Welcome Music ----------
# You can replace the file with your own mp3 placed in project root
def get_audio_base64(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    audio_base64 = get_audio_base64("welcome_music.mp3")

    st.markdown(f"""
    <audio autoplay loop hidden>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

except:
    pass  # If music file not present, skip silently


# ---------- Header ----------
st.markdown("""
<div style="
background: linear-gradient(90deg, #00F5FF, #3A86FF);
padding:30px;
border-radius:14px;
color:white;
text-align:center;
font-size:36px;
font-weight:bold;
margin-bottom:50px;
box-shadow:0px 6px 25px rgba(0,245,255,0.5);
">
🧠 Supervised Machine Learning Studio - End to End Machine Learning Platform
</div>
""", unsafe_allow_html=True)

# ---------- Welcome Box ----------
st.markdown("""
<div style="
background: linear-gradient(135deg, #FFFFFF 0%, #FFFDF4 40%, #F6E7B4 75%, #EFD98A 100%);
padding:45px;
border-radius:18px;
border: 1px solid rgba(239, 217, 138, 0.6);
box-shadow:0px 8px 30px rgba(239, 217, 138, 0.35);
max-width:850px;
margin:0 auto 60px auto;
text-align:center;
">
<div style="font-size:28px; font-weight:600; color:#0A1F44;">
Welcome to Supervised ML Studio
</div>
<div style="font-size:18px; color:#2F3E5C; margin-top:15px;">
Build complete Machine Learning pipelines with full control.
</div>
</div>
""", unsafe_allow_html=True)

# ---------- FLOW SECTION ----------
steps = [
    ("📂 Upload Data", "#E3F2FD", "#BBDEFB"),
    ("🧹 Clean", "#E8F5E9", "#C8E6C9"),
    ("🔄 Transform", "#FFF3E0", "#FFE0B2"),
    ("🤖 Train", "#F3E5F5", "#E1BEE7"),
    ("📊 Evaluate", "#E0F7FA", "#B2EBF2"),
    ("🎯 Predict", "#FFFDE7", "#FFF9C4"),
]

flow_html = '<div class="flow-container">'

for i, (text, c1, c2) in enumerate(steps):
    flow_html += f"""
    <div class="flow-box" style="background:linear-gradient(135deg,{c1},{c2});">
        {text}
    </div>
    """

    if i < len(steps) - 1:
        flow_html += '<div class="arrow">➜</div>'

flow_html += "</div>"

st.markdown(flow_html, unsafe_allow_html=True)

st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)

# ---------- Navigation Button ----------
col1, col2, col3 = st.columns([3,2,3])

with col2:
    if st.button("🚀 Start with Data Upload", use_container_width=True):
        st.switch_page("pages/1_Upload_data.py")