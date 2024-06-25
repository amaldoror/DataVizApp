import streamlit as st


show_latex = st.checkbox("Show LaTeX")
# LaTeX Button
if show_latex:
    st.latex(r'''
        a + ax + a x^2 + a x^3 + \cdots + a x^{n-1} =
        \sum_{k=0}^{n-1} ax^k =
        a \left(\frac{1-x^{n}}{1-x}\right)
        ''')

