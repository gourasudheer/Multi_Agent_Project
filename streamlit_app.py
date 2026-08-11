import streamlit as st

from agentic_ai_blog import ConditionalAgenticWorkflow, OpenAIClient

st.set_page_config(
    page_title="Agentic AI Conditional Workflow",
    page_icon="🤖",
    layout="wide",
)


def clear_chat() -> None:
    st.session_state.prompt = ""
    st.session_state.final_response = None
    st.session_state.route = None


def main() -> None:
    st.title("Agentic AI Conditional Workflow")
    st.write(
        "Enter a request and the agent router will select the best specialist: chat, code, or writer."
    )

    with st.expander("How it works", expanded=False):
        st.write(
            "This app routes your request to the best available agent and returns the final response quickly."
        )

    # initialize session state
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "final_response" not in st.session_state:
        st.session_state.final_response = None
    if "route" not in st.session_state:
        st.session_state.route = None

    prompt = st.text_area("Request or topic", height=220, key="prompt")
    show_route = st.checkbox("Show selected route", value=True, key="show_route")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate response"):
            if not st.session_state.prompt or not st.session_state.prompt.strip():
                st.warning("Please enter a prompt to continue.")
            else:
                try:
                    client = OpenAIClient()
                    workflow = ConditionalAgenticWorkflow(client)

                    with st.spinner("Routing and generating response..."):
                        result = workflow.generate_response(st.session_state.prompt)

                    st.session_state.final_response = result.final_response
                    st.session_state.route = result.route

                except Exception as exc:
                    st.session_state.final_response = None
                    st.error(f"Failed to generate the response: {exc}")

    with col2:
        st.button("Clear chat", on_click=clear_chat)

    if st.session_state.route and show_route:
        st.success(f"Selected route: {st.session_state.route}")

    if st.session_state.final_response:
        st.subheader("Final Response")
        st.write(st.session_state.final_response)


if __name__ == "__main__":
    main()
