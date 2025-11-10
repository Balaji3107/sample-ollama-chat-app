# claims_system_ai_chatlit.py
import os
import httpx
import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # type: ignore
import uuid

# ---------------------- CONFIG ----------------------
os.environ["TIKTOKEN_CACHE_DIR"] = "c:/tiktoken_cache"
client = httpx.Client(verify=False)

# LLM instance
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-4o",
    api_key="sk-GB2xk_Me79xnRTOzilZi2Q",
    http_client=client,
)

# Embedding model
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-GB2xk_Me79xnRTOzilZi2Q",
    http_client=client,
)

# Chroma DB setup
client_chroma = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-GB2xk_Me79xnRTOzilZi2Q",
    model_name="azure/genailab-maas-text-embedding-3-large",
)
policy_collection = client_chroma.get_or_create_collection(
    name="policies", embedding_function=embedding_fn
)
claim_collection = client_chroma.get_or_create_collection(
    name="claims", embedding_function=embedding_fn
)

# ---------------------- LOGIN SCREEN ----------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

users = {
    'admin': 'admin123',
    'branch_manager': 'manager123'
}

roles = {
    'admin': 'Admin',
    'branch_manager': 'Branch Manager'
}

if not st.session_state.logged_in:
    st.title('🔒 Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.role = roles[username]
            st.success(f'Logged in as {st.session_state.role}')
            st.experimental_rerun()
        else:
            st.error('Invalid username or password')
    st.stop()
else:
    st.sidebar.write(f'Logged in as: {st.session_state.role}')

# ---------------------- AI AGENTS ----------------------
def document_extraction_agent(claim_text):
    """Simulated OCR/NLP extraction"""
    # Here, you could use llm.invoke for smarter extraction
    resp = llm.invoke(f"Extract structured info from this claim: {claim_text}")
    return resp.content

def fraud_detection_agent(extracted_data):
    """Simulated fraud detection using anomaly detection"""
    # Dummy logic: flag if claim mentions 'fire' or 'accident'
    flag = any(x in extracted_data.lower() for x in ["fire", "accident"])
    return flag

def damage_assessment_agent(claim_text):
    """Simulated damage assessment"""
    # Return a damage score from 0-100
    resp = llm.invoke(f"Assess damage severity in this claim: {claim_text}")
    try:
        score = int(''.join(filter(str.isdigit, resp.content)))
    except:
        score = 50  # default
    return min(max(score, 0), 100)

def payout_calculation_agent(policy_details, damage_score):
    """Calculate payout based on policy terms and damage score"""
    # Assume policy_details has 'coverage_limit' and 'deductible'
    limit = int(policy_details.get("coverage_limit", 10000))
    deductible = int(policy_details.get("deductible", 1000))
    payout = max(0, (damage_score / 100 * limit) - deductible)
    return payout

def customer_communication_agent(claim_id, status):
    """Simulated notification"""
    return f"Notification sent for claim {claim_id} with status '{status}'"

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="AI Claims Processing", layout="wide")
st.title("🧠 End-to-End AI Claims Processing")

# ---------------------- Policy Management ----------------------
st.subheader("1️⃣ Create New Policy")
policy_name = st.text_input("Policy Name")
coverage_limit = st.text_input("Coverage Limit", value="10000")
deductible = st.text_input("Deductible", value="1000")

if st.button("Create Policy"):
    if policy_name:
        policy_id = str(uuid.uuid4())
        policy_collection.add(
            documents=[policy_name],
            metadatas=[{"coverage_limit": coverage_limit, "deductible": deductible}],
            ids=[policy_id],
        )
        st.success(f"Policy '{policy_name}' created with ID {policy_id}")
    else:
        st.error("Policy name cannot be empty")

# Fetch available policies
policies = policy_collection.get(include_metadata=True)
policy_options = [
    (p["id"], p["document"], p["metadata"]) for p in zip(policies["ids"], policies["documents"], policies["metadatas"])
]

# ---------------------- Claim Management ----------------------
st.subheader("2️⃣ Create New Claim")
if policy_options:
    selected_policy = st.selectbox(
        "Select Policy", options=[f"{p[1]} ({p[0]})" for p in policy_options]
    )
    claim_text = st.text_area("Enter Claim Details")

    if st.button("Submit Claim"):
        if claim_text.strip():
            policy_id = selected_policy.split("(")[-1].strip(")")
            policy_metadata = next(p[2] for p in policy_options if p[0] == policy_id)

            extracted_data = document_extraction_agent(claim_text)
            fraud_flag = fraud_detection_agent(extracted_data)
            damage_score = damage_assessment_agent(claim_text)
            payout = payout_calculation_agent(policy_metadata, damage_score)

            claim_id = str(uuid.uuid4())
            claim_collection.add(
                documents=[claim_text],
                metadatas=[
                    {
                        "policy_id": policy_id,
                        "claim_text": claim_text,
                        "extracted_data": extracted_data,
                        "fraud_flag": fraud_flag,
                        "damage_score": damage_score,
                        "payout_amount": payout,
                        "status": "Submitted",
                    }
                ],
                ids=[claim_id],
            )

            notification = customer_communication_agent(claim_id, "Submitted")
            st.success(f"Claim submitted successfully! {notification}")
        else:
            st.error("Claim details cannot be empty")
else:
    st.warning("No policies available. Create a policy first!")

# ---------------------- View Claims ----------------------
st.subheader("3️⃣ View All Claims")
claims = claim_collection.get(include_metadata=True)
if claims["ids"]:
    data = []
    for doc_id, meta in zip(claims["ids"], claims["metadatas"]):
        data.append(
            {
                "Claim ID": doc_id,
                "Policy ID": meta.get("policy_id"),
                "Claim Text": meta.get("claim_text"),
                "Extracted Data": meta.get("extracted_data"),
                "Fraud Flag": meta.get("fraud_flag"),
                "Damage Score": meta.get("damage_score"),
                "Payout Amount": meta.get("payout_amount"),
                "Status": meta.get("status"),
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df)
else:
    st.info("No claims found yet.")
