import streamlit as st

from utils import get_image

IMAGE_WIDTH = 256
user_avatar, diffusion_model_avatar = "🧑🏻‍💼", "⚡️"
custom_css = """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
    }
    .st-emotion-cache-janbn0 p {
        text-align: right;
        margin-right: 10px;
    }
    .st-emotion-cache-q58vwc p {
        text-align:right;
    }
    .st-emotion-cache-k7vsyb h1{
        text-align:center;
        margin-top: -60px;
    }  
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        # Photo Generator 📸
        ### *User Guide*
            1. Snap or select a selfie
            2. Craft your prompt
            3. Hit send
            4. Have fun!
        """
    )
    st.caption("Created by @atadanicen")
    st.divider()

    negative_prompt_on = st.toggle("Negative Prompt")
    negative_prompt = (
        st.text_area("Enter Your Negative Prompt") if negative_prompt_on else ""
    )

    image_input = st.radio("Visual Input Choice", ("Upload", "Camera"), horizontal=True)
    if image_input == "Camera":
        image = st.camera_input("Take A Selfie")
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        image = uploaded_file.read() if uploaded_file else None
        if uploaded_file:
            st.image(uploaded_file)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    if message["role"] == "diffusion_model":
        with st.chat_message("diffusion_model", avatar=message["avatar"]):
            st.image(message["image"], width=IMAGE_WIDTH, caption=message["prompt"])
            history_download_button = st.download_button(
                "Download Image",
                message["image"],
                file_name="generated_image.png",
                key=message["key"],
            )
            if history_download_button:
                st.toast("Generated image was saved!", icon="📸")
    else:
        st.chat_message("user", avatar=message["avatar"]).write(message["prompt"])

prompt = st.chat_input("Enter Your Prompt")

if prompt and image:
    with st.spinner(text="Image Generating..."):
        try:
            image_created = get_image(prompt, negative_prompt, image)
            if image_created:
                # Display user prompt
                user_chat = st.chat_message("user", avatar=user_avatar)
                st.session_state.messages.append(
                    {"role": "user", "prompt": prompt, "avatar": user_avatar}
                )

                # Display generated image and download button
                diffusion_model_chat = st.chat_message(
                    "diffusion_model", avatar=diffusion_model_avatar
                )
                download_button_key = f"{prompt}_{len(st.session_state.messages) // 2}"
                st.session_state.messages.append(
                    {
                        "role": "diffusion_model",
                        "avatar": diffusion_model_avatar,
                        "prompt": prompt,
                        "image": image_created,
                        "key": download_button_key,
                    }
                )
                st.rerun()
        except Exception:
            st.warning("API Connection Error")
