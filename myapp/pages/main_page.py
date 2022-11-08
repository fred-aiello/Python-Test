# Contents of ~/my_app/main_page.py
import streamlit as st

st.markdown("# Main page  🎈")
st.sidebar.markdown("# Main page side menu 🎈")

expander=st.expander("Interesting hyperlinks")

expander.write("check out this [Streamlit Installation](https://docs.streamlit.io/library/get-started/multipage-apps/create-a-multipage-app) it's test")

expander.write("check out this [Multipages](https://docs.streamlit.io/library/get-started/installation)")
expander.image("https://static.streamlit.io/examples/dice.jpg")



