
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
import tempfile
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Flexure Analysis", layout="wide", page_icon="🏗️")

# Dark Theme & Custom CSS
plt.style.use('dark_background')
ST_BG_COLOR = "#0e1117"
st.markdown(f"""
    <style>
    .stApp {{ background-color: {ST_BG_COLOR}; }}
    div.stButton > button:first-child {{
        background-color: #007acc;
        color: white;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }}
    div.stButton > button:hover {{
        background-color: #005f9e;
        color: white;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

# Visual Constants
C_BG_MAIN    = "#1e1e1e"
G_BEAM       = "#9e9e9e"      
G_SUP        = "#b0bec5"      
G_POINT      = "#ff5252"      
G_MOMENT     = "#d500f9"      
G_UDL_LINE   = "#4fc3f7"      
G_UDL_FILL   = "#03a9f4"      
G_UVL_LINE   = "#ffb74d"      
G_UVL_FILL   = "#ff9800"      

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'loads' not in st.session_state:
    st.session_state.loads = []
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# ==========================================
# 3. PDF REPORT GENERATOR (WITH IMAGES)
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Flexure: Engineering Analysis Report', border=False, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf(L, beam_type, loads, Ra, Rb, Ma, max_shear, max_moment, fig_preview, fig_graphs):
    # Initialize PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    # --- PAGE 1: TEXT RESULTS ---
    
    # 1. System Properties
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "1. System Configuration", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 8, f"Beam Type: {beam_type}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Length: {L} m", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 2. Load List
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "2. Applied Loads", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    
    # Table Header
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(30, 8, "Type", border=1, align='C', fill=True)
    pdf.cell(40, 8, "Magnitude", border=1, align='C', fill=True)
    pdf.cell(50, 8, "Location/Range", border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
    
    # Table Rows
    pdf.set_font("helvetica", size=10)
    for l in loads:
        type_str = l["type"]
        mag_str = ""
        loc_str = ""
        
        if l["type"] == "POINT":
            mag_str = f"{l['mag']} kN"
            loc_str = f"at {l['loc']} m"
        elif l["type"] == "MOMENT":
            mag_str = f"{l['mag']} kNm"
            loc_str = f"at {l['loc']} m"
        elif l["type"] == "UDL":
            mag_str = f"{l['mag']} kN/m"
            loc_str = f"{l['start']} - {l['end']} m"
        elif l["type"] == "UVL":
            mag_str = f"{l['start_mag']} - {l['end_mag']} kN/m"
            loc_str = f"{l['start']} - {l['end']} m"
            
        pdf.cell(30, 8, type_str, border=1)
        pdf.cell(40, 8, mag_str, border=1)
        pdf.cell(50, 8, loc_str, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # 3. Calculated Reactions
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "3. Equilibrium Results", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 8, f"Vertical Reaction at A (Ra): {Ra:.3f} kN", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Vertical Reaction at B (Rb): {Rb:.3f} kN", new_x="LMARGIN", new_y="NEXT")
    if Ma != 0:
        pdf.cell(0, 8, f"Wall Moment at A (Ma): {Ma:.3f} kNm", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 8, "Wall Moment at A (Ma): 0.000 kNm", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 4. Critical Design Values
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "4. Critical Design Values", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    # Highlight critical values
    pdf.set_text_color(200, 0, 0) # Red
    pdf.cell(0, 8, f"Absolute Max Shear Force: {abs(max_shear):.3f} kN", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Absolute Max Bending Moment: {abs(max_moment):.3f} kNm", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0) # Reset to black

    # --- PAGE 2: DIAGRAMS ---
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "5. Free Body & Analysis Diagrams", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # We need to save the Matplotlib figures to temp files to embed them
    # Note: We use a dark theme in the app, but for PDF report we usually want
    # light theme. However, converting themes on the fly is tricky.
    # For now, we print the dark theme plots which look cool, or we can force a light save.
    
    # --- 1. Embed Free Body Diagram (Preview) ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile_preview:
        # Save figure to temp file
        fig_preview.savefig(tmpfile_preview.name, dpi=150, bbox_inches='tight', facecolor=C_BG_MAIN)
        # Embed in PDF (x, y, width)
        pdf.image(tmpfile_preview.name, x=10, y=None, w=190)
        preview_path = tmpfile_preview.name

    pdf.ln(5)
    
    # --- 2. Embed SFD & BMD ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile_graphs:
        fig_graphs.savefig(tmpfile_graphs.name, dpi=150, bbox_inches='tight', facecolor=C_BG_MAIN)
        pdf.image(tmpfile_graphs.name, x=10, y=None, w=190)
        graphs_path = tmpfile_graphs.name

    # Clean up temp files
    try:
        os.remove(preview_path)
        os.remove(graphs_path)
    except:
        pass

    return bytes(pdf.output())

# ==========================================
# 4. SIDEBAR - CONTROLS
# ==========================================
with st.sidebar:
    st.title("🏗️ Flexure")
    st.caption("Professional Beam Analysis")
    
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
            st.session_state.analyzed = False

    with tab2:
        m_mag = st.number_input("Moment Mag (kNm)", value=10.0, key="m_m")
        m_loc = st.number_input("Moment Loc (m)", value=5.0, key="m_l")
        if st.button("Add Moment"):
            st.session_state.loads.append({"type": "MOMENT", "mag": m_mag, "loc": m_loc})
            st.session_state.analyzed = False

    with tab3:
        u_mag = st.number_input("UDL Mag (kN/m)", value=5.0, key="u_m")
        u_s = st.number_input("Start (m)", value=2.0, key="u_s")
        u_e = st.number_input("End (m)", value=8.0, key="u_e")
        if st.button("Add UDL"):
            st.session_state.loads.append({"type": "UDL", "mag": u_mag, "start": u_s, "end": u_e})
            st.session_state.analyzed = False

    with tab4:
        v_s_mag = st.number_input("Start Mag", value=0.0, key="v_sm")
        v_e_mag = st.number_input("End Mag", value=10.0, key="v_em")
        v_s = st.number_input("Start (m)", value=2.0, key="v_s")
        v_e = st.number_input("End (m)", value=8.0, key="v_e")
        if st.button("Add UVL"):
            st.session_state.loads.append({"type": "UVL", "start_mag": v_s_mag, "end_mag": v_e_mag, "start": v_s, "end": v_e})
            st.session_state.analyzed = False

    st.divider()
    
    if st.session_state.loads:
        st.write("Current Loads:")
        for i, l in enumerate(st.session_state.loads):
            st.caption(f"{i+1}. {l['type']} | {l}")
        
        if st.button("Clear All Loads"):
            st.session_state.loads = []
            st.session_state.analyzed = False
            st.rerun()
            
    st.divider()
    if st.button("▶ RUN ANALYSIS"):
        st.session_state.analyzed = True

# ==========================================
# 5. MATH ENGINE
# ==========================================
def solve_beam():
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
        if denom == 0: denom = 0.0001
        Rb = m_sum_about_A / denom
        Ra = f_sum_vertical - Rb
        Ma = 0

    x_vals = np.linspace(0, L, 500)
    V_vals = []
    M_vals = []
    
    for x in x_vals:
        v = 0; m = 0
        
        if x > SupA:
            v += Ra; m += Ra * (x - SupA) - Ma
        if beam_type != "Cantilever" and x > SupB:
            v += Rb; m += Rb * (x - SupB)
            
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
# 6. VISUALIZATION
# ==========================================
st.subheader("1. Load Diagram (Preview)")
fig_preview, ax0 = plt.subplots(figsize=(10, 3))
fig_preview.patch.set_facecolor(ST_BG_COLOR)
ax0.set_facecolor(ST_BG_COLOR)
ax0.axis('off')
ax0.set_xlim(-L*0.1, L*1.1); ax0.set_ylim(-2.5, 5.0)

ax0.plot([0, L], [0, 0], color=G_BEAM, lw=6, solid_capstyle='round', zorder=2)

if beam_type == "Cantilever":
    ax0.plot([SupA, SupA], [-1.5, 1.5], color=G_SUP, lw=6, zorder=1)
else:
    ax0.plot(SupA, -0.35, marker='^', markersize=14, color=G_SUP, mec=ST_BG_COLOR, zorder=3)
    ax0.plot(SupB, -0.35, marker='o', markersize=12, color=G_SUP, mec=ST_BG_COLOR, zorder=3)

for l in st.session_state.loads:
    if l["type"] == "POINT":
        ax0.arrow(l["loc"], 2.0, 0, -1.6, head_width=L*0.025, length_includes_head=True, fc=G_POINT, ec=G_POINT, zorder=5)
        ax0.text(l["loc"], 2.2, f"{l['mag']}kN", ha='center', color=G_POINT, fontweight='bold', fontsize=10)
    elif l["type"] == "MOMENT":
        symbol = r"$\circlearrowright$" if l["mag"] >= 0 else r"$\circlearrowleft$"
        ax0.text(l["loc"], 0, symbol, ha='center', va='center', color=G_MOMENT, fontsize=24, zorder=5)
        ax0.text(l["loc"], 0.6, f"{l['mag']}kNm", ha='center', color=G_MOMENT, fontweight='bold', fontsize=10)
    elif l["type"] == "UDL":
        ax0.fill_between([l["start"], l["end"]], 0.3, 1.0, color=G_UDL_FILL, alpha=0.3)
        ax0.plot([l["start"], l["end"]], [1.0, 1.0], color=G_UDL_LINE, lw=2)
        ax0.text((l["start"]+l["end"])/2, 1.2, f"{l['mag']} kN/m", ha='center', color=G_UDL_LINE, fontweight='bold', fontsize=10)
    elif l["type"] == "UVL":
        h1 = 0.3 + (l["start_mag"]/50); h2 = 0.3 + (l["end_mag"]/50)
        ax0.fill([l["start"], l["end"], l["end"], l["start"]], [0.3, 0.3, h2, h1], color=G_UVL_FILL, alpha=0.4)
        mid = (l["start"]+l["end"])/2
        ax0.text(mid, max(h1,h2)+0.2, f"{l['start_mag']}→{l['end_mag']}", ha='center', color=G_UVL_LINE, fontweight='bold', fontsize=10)

st.pyplot(fig_preview)

if st.session_state.analyzed:
    st.divider()
    st.subheader("2. Analysis Results")
    
    x, V, M, Ra, Rb, Ma = solve_beam()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reaction A", f"{Ra:.2f} kN")
    c2.metric("Reaction B", f"{Rb:.2f} kN")
    
    mx = 0
    if len(M) > 0: mx = max(M, key=abs)
    c3.metric("Max Moment", f"{mx:.2f} kNm")
    
    if Ma != 0: c4.metric("Wall Moment", f"{Ma:.2f} kNm")

    fig_graphs, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig_graphs.patch.set_facecolor(ST_BG_COLOR)
    plt.subplots_adjust(hspace=0.4)

    ax1.set_facecolor(ST_BG_COLOR)
    ax1.plot(x, V, color=G_UDL_LINE, lw=2)
    ax1.fill_between(x, V, 0, color=G_UDL_FILL, alpha=0.2)
    ax1.set_ylabel("Shear (kN)", color="white")
    ax1.set_title("Shear Force Diagram", color="white")
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines['bottom'].set_color('white'); ax1.spines['top'].set_color('#333')
    ax1.spines['left'].set_color('white'); ax1.spines['right'].set_color('#333')

    ax2.set_facecolor(ST_BG_COLOR)
    ax2.plot(x, M, color=G_UVL_LINE, lw=2)
    ax2.fill_between(x, M, 0, color=G_UVL_FILL, alpha=0.2)
    ax2.set_ylabel("Moment (kNm)", color="white")
    ax2.set_title("Bending Moment Diagram", color="white")
    ax2.set_xlabel("Length (m)", color="white")
    ax2.tick_params(colors='white')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['bottom'].set_color('white'); ax2.spines['top'].set_color('#333')
    ax2.spines['left'].set_color('white'); ax2.spines['right'].set_color('#333')
    
    if len(M)>0:
        mx_loc = x[np.argmax(np.abs(M))]
        ax2.plot(mx_loc, mx, 'o', color='white', markersize=6)
        ax2.text(mx_loc, mx, f" {mx:.2f}", color='white', fontsize=10, va='bottom')

    st.pyplot(fig_graphs)

    # --- DOWNLOAD SECTION ---
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    
    # 1. CSV Download
    df = pd.DataFrame({"Length (m)": x, "Shear (kN)": V, "Moment (kNm)": M})
    csv = df.to_csv(index=False).encode('utf-8')
    col_dl1.download_button("📥 Download CSV Data", csv, "flexure_data.csv", "text/csv")
    
    # 2. PDF Report Download
    max_shear = max(V, key=abs) if len(V) > 0 else 0
    # Pass BOTH figures (fig_preview and fig_graphs) to the PDF generator
    pdf_bytes = generate_pdf(L, beam_type, st.session_state.loads, Ra, Rb, Ma, max_shear, mx, fig_preview, fig_graphs)
    col_dl2.download_button("📄 Download PDF Report", pdf_bytes, "flexure_report.pdf", "application/pdf")

else:
    st.info("👈 Add loads in the sidebar and click 'RUN ANALYSIS' to see results.")
