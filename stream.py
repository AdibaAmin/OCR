import streamlit as st
from PIL import Image
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import io
import pytesseract
from dotenv import load_dotenv
import os


load_dotenv()
google_api_key = os.getenv("GOOGL_API_KEY")


llm = ChatGoogleGenerativeAI(
    google_api_key=google_api_key,
    temperature=0.3,
    model="gemini-1.5-flash-latest"
)


def extract_text_from_file(uploaded_file):
    try:
        image = Image.open(uploaded_file)


        gray_image = image.convert("L")

        extracted_text = pytesseract.image_to_string(gray_image).strip()
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None


st.set_page_config(page_title="Invoice Information Extractor", layout="wide")
st.title("📑 Invoice Information Extractor")

uploaded_file = st.file_uploader("Upload an Invoice Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Invoice", use_column_width=True)

    if st.button("Extract Information"):
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_file(uploaded_file)

        if extracted_text:
            st.subheader("🔎 Extracted Text:")
            st.text(extracted_text)

            st.subheader("🤖 Processing with Gemini...")
            message = f"""
            system: You are an invoice information extractor who extracts information from text and converts it into a JSON format with proper structure and key-value pairs.
            user: {extracted_text}
            """

            try:
                response = llm.invoke(message)
                result = str(response.content)
                result = result.replace("```json", "").replace("```", "").strip()

                json_data = json.loads(result)

                st.subheader("📋 Extracted Invoice Details (JSON):")
                st.json(json_data)

                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(json_data, indent=4),
                    file_name="invoice_data.json",
                    mime="application/json"
                )

            except json.JSONDecodeError:
                st.error("Invalid JSON response from AI. Please check the model output.")
            except Exception as e:
                st.error(f"Error processing invoice: {str(e)}")
