import streamlit as st
from PyPDF2 import PdfFileReader

with st.sidebar:
    layer = st.selectbox(
        label="Select layer",
        options=[
            # "inception3a",
            "inception3b",
            "inception4a",
            "inception4b",
            "inception4c",
            "inception4d",
            "inception4e",
            "inception5a",
            "inception5b",
        ],
    )

pdf = PdfFileReader(f"pdf_files/Test_html_{layer}.pdf")
for i in range(pdf.getNumPages()):
    st.title("Cluster 1")
