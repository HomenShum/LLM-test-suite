"""Footer rendering extracted from the main app."""

import streamlit as st

def render_footer() -> None:
    with st.expander("Download options & CLI command", expanded=False):
        show_previews = st.session_state.get("show_main_df_previews", False)
        if show_previews and "df" in st.session_state:
            st.divider()
            st.subheader("Current DataFrame State")
            display_df = st.session_state.df.copy()
            rationale_cols = [c for c in display_df.columns if c.endswith("_rationale") or c == "judge_rationale"]
            for col in rationale_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str).str.slice(0, 150) + "..."
            st.dataframe(display_df, use_container_width=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "???,? Download Results as CSV",
                    data=csv_bytes,
                    file_name="classification_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                pass

        st.divider()
        st.caption("Live App: https://llm-test-suite-cafecorner.streamlit.app/")

        st.code("streamlit run streamlit_app.py", language="bash")
