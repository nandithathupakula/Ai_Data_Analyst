import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ollama

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst Chatbot (Local AI)")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Column selectors for graphs
    st.subheader("📊 Graph Settings")
    x_col = st.selectbox("Select X-axis", df.columns)
    y_col = st.selectbox("Select Y-axis", df.columns)

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your data...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # 🔥 GRAPH DETECTION LOGIC
        if any(word in user_input.lower() for word in ["bar graph", "plot", "chart", "graph"]):
            with st.chat_message("assistant"):
                st.write("📊 Generating graph...")

                try:
                    plt.figure()
                    plt.bar(df[x_col], df[y_col])
                    plt.xticks(rotation=45)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f"{y_col} vs {x_col}")

                    st.pyplot(plt)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Generated a bar graph of {y_col} vs {x_col}"
                    })

                except Exception as e:
                    st.write("Error generating graph:", e)

        else:
            # Normal AI response
            data_sample = df.head(20).to_string()

            # Include conversation history
            conversation = ""
            for msg in st.session_state.messages:
                conversation += f"{msg['role']}: {msg['content']}\n"

            prompt = f"""
            You are an expert data analyst.

            Dataset:
            {data_sample}

            Conversation:
            {conversation}

            Answer the latest question clearly and concisely.
            """

            try:
                response = ollama.chat(
                    model='llama3',
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response['message']['content']

            except Exception as e:
                answer = "⚠️ Error connecting to Ollama. Make sure it is running."

            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.write(answer)