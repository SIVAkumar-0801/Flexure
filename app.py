import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Beam Studio Web", layout="wide", page_icon="🏗️")

# Dark Theme adjustments for Matplotlib
plt.style.use('dark_background')
ST_BG_COLOR = "#0e1117"
st.markdown(f"""
    <style>
    .stApp {{ background-color: {ST_BG_COLOR}; }}
    </style>
    """, unsafe_allow_html=True)

# Graph Colors
G_UDL_LINE = "#4fc3f7"
G_UDL_FILL = "#03a9f4"
G_UVL_LINE = "#ffb74d"
G_UVL_FILL = "#ff9800"

# ==========================================
# 2. SESSION STATE (Memory)
# ==========================================
if 'loads' not in st.session_state:
    st.session_state.loads = []

# ==========================================
# 3. SIDEBAR - CONTROLS
# ==========================================
with st.sidebar:
    st.title("🏗️ Beam Studio")
    st.caption("Professional Web Edition")
    
    st.header("1. Structure")
    beam_type = st.selectbox("Support Type", ["Simply Supported", "Cantilever"])
    L = st.number_input("Length (m)", value=10.0, min_value=0.1)
    
    SupA = st.number_input("Support A Location (m)", value=0.0)
    SupB = 10.0
    if beam_type == "Simply Supported":
        SupB = st.number_input("Support B Location (m)", value=10.0)
    
    st.divider()
    
    st.header("2. Add Loads")
    tab1, tab2, tab3, tab4 = st.tabs(["Point", "Moment", "UDL", "UVL"])
    
    with tab1:
        p_mag = st.number_input("Point Mag (kN)", value=10.0, key="p_m")
        p_loc = st.number_input("Point Loc (m)", value=5.0, key="p_l")
        if st.button("Add Point Load"):
            st.session_state.loads.append({"type": "POINT", "mag": p_mag, "loc": p_loc})

    with tab2:
        m_mag = st.number_input("Moment Mag (kNm)", value=10.0, key="m_m")
        m_loc = st.number_input("Moment Loc (m)", value=5.0, key="m_l")
        if st.button("Add Moment"):
            st.session_state.loads.append({"type": "MOMENT", "mag": m_mag, "loc": m_loc})

    with tab3:
        u_mag = st.number_input("UDL Mag (kN/m)", value=5.0, key="u_m")
        u_s = st.number_input("Start (m)", value=2.0, key="u_s")
        u_e = st.number_input("End (m)", value=8.0, key="u_e")
        if st.button("Add UDL"):
            st.session_state.loads.append({"type": "UDL", "mag": u_mag, "start": u_s, "end": u_e})

    with tab4:
        v_s_mag = st.number_input("Start Mag", value=0.0, key="v_sm")
        v_e_mag = st.number_input("End Mag", value=10.0, key="v_em")
        v_s = st.number_input("Start (m)", value=2.0, key="v_s")
        v_e = st.number_input("End (m)", value=8.0, key="v_e")
        if st.button("Add UVL"):
            st.session_state.loads.append({"type": "UVL", "start_mag": v_s_mag, "end_mag": v_e_mag, "start": v_s, "end": v_e})

    st.divider()
    st.header("Current Loads")
    if st.session_state.loads:
        for i, l in enumerate(st.session_state.loads):
            st.text(f"{i+1}. {l['type']} | {l}")
        
        if st.button("Clear All Loads"):
            st.session_state.loads = []
            st.rerun()

# ==========================================
# 4. MAIN LOGIC ENGINE (Reused from your code)
# ==========================================
def solve_beam():
    # Setup Reactions
    m_sum_about_A = 0 
    f_sum_vertical = 0
    
    for l in st.session_state.loads:
        if l["type"] == "POINT":
            f = l["mag"]
            m_sum_about_A += f * (l["loc"] - SupA)
            f_sum_vertical += f
        elif l["type"] == "MOMENT":
            m_sum_about_A += l["mag"]
        elif l["type"] == "UDL":
            length = l["end"] - l["start"]
            f = l["mag"] * length
            centroid = l["start"] + (length / 2) - SupA
            m_sum_about_A += f * centroid
            f_sum_vertical += f
        elif l["type"] == "UVL":
            w1, w2 = l["start_mag"], l["end_mag"]
            length = l["end"] - l["start"]
            F1 = min(w1, w2) * length
            F2 = 0.5 * abs(w2 - w1) * length
            c1 = l["start"] + (length / 2) - SupA
            c2 = l["start"] + (2/3 if w2 > w1 else 1/3)*length - SupA
            m_sum_about_A += (F1 * c1) + (F2 * c2)
            f_sum_vertical += (F1 + F2)

    if beam_type == "Cantilever":
        Rb = 0
        Ra = f_sum_vertical
        Ma = m_sum_about_A
    else:
        denom = (SupB - SupA)
        if denom == 0: denom = 0.0001 # Prevent div by zero
        Rb = m_sum_about_A / denom
        Ra = f_sum_vertical - Rb
        Ma = 0

    # Method of Sections
    x_vals = np.linspace(0, L, 500)
    V_vals = []
    M_vals = []
    
    for x in x_vals:
        v = 0; m = 0
        
        # Supports
        if x > SupA:
            v += Ra; m += Ra * (x - SupA) - Ma
        if beam_type != "Cantilever" and x > SupB:
            v += Rb; m += Rb * (x - SupB)
            
        # Loads
        for l in st.session_state.loads:
            if l["type"] == "POINT" and x > l["loc"]:
                v -= l["mag"]; m -= l["mag"] * (x - l["loc"])
            elif l["type"] == "MOMENT" and x > l["loc"]:
                m -= l["mag"]
            elif l["type"] == "UDL" and x > l["start"]:
                ov = min(x, l["end"]) - l["start"]
                f = l["mag"] * ov
                v -= f; m -= f * (x - (l["start"] + ov / 2))
            elif l["type"] == "UVL" and x > l["start"]:
                curr_end = min(x, l["end"])
                dx = curr_end - l["start"]
                w1 = l["start_mag"]
                w_at_x = w1 + ((l["end_mag"] - w1) * (dx / (l["end"] - l["start"])))
                f_slice = ((w1 + w_at_x) / 2) * dx
                cent = (dx / 3) * ((w1 + 2 * w_at_x) / (w1 + w_at_x))
                v -= f_slice; m -= f_slice * (x - (l["start"] + cent))
        
        V_vals.append(v)
        M_vals.append(m)
        
    return x_vals, V_vals, M_vals, Ra, Rb, Ma

# ==========================================
# 5. VISUALIZATION
# ==========================================
x, V, M, Ra, Rb, Ma = solve_beam()

# --- Top Section: Key Values ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reaction A", f"{Ra:.2f} kN")
col2.metric("Reaction B", f"{Rb:.2f} kN")
max_m = max(M, key=abs) if M else 0
col3.metric("Max Moment", f"{max_m:.2f} kNm")
if Ma != 0: col4.metric("Wall Moment", f"{Ma:.2f} kNm")

# --- Graphs ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.patch.set_facecolor(ST_BG_COLOR)

# SFD
ax1.set_facecolor(ST_BG_COLOR)
ax1.plot(x, V, color=G_UDL_LINE, lw=2)
ax1.fill_between(x, V, 0, color=G_UDL_FILL, alpha=0.2)
ax1.set_ylabel("Shear Force (kN)", color="white")
ax1.set_title("Shear Force Diagram", color="white")
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.grid(True, linestyle='--', alpha=0.3)

# BMD
ax2.set_facecolor(ST_BG_COLOR)
ax2.plot(x, M, color=G_UVL_LINE, lw=2)
ax2.fill_between(x, M, 0, color=G_UVL_FILL, alpha=0.2)
ax2.set_ylabel("Bending Moment (kNm)", color="white")
ax2.set_title("Bending Moment Diagram", color="white")
ax2.set_xlabel("Length (m)", color="white")
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
ax2.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig)

# --- CSV Download ---
df = pd.DataFrame({"Length (m)": x, "Shear (kN)": V, "Moment (kNm)": M})
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Results CSV", csv, "beam_analysis.csv", "text/csv")
