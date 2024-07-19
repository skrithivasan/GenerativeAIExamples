import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

def app():
    st.title("Visualize the Knowledge Graph!")
    st.subheader("Load a knowledge graph GraphML file from your system.")
    st.write("If you used the previous step, it will be saved on your system as ```knowledge_graph.graphml```")
    
    components.iframe(
        src="https://gephi.org/gephi-lite/",
        height=800,
        scrolling=True,
    )

if __name__ == "__main__":
    app()
