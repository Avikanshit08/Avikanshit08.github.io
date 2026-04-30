import CodeAwareTextSplitter

import streamlit as st
from langchain_openai import *
from langchain_community.vectorstores import *
from langchain_core.documents import *
import httpx, zipfile, tempfile, requests, os, base64

CODE_EXTENSIONS = (
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".sh",
    ".sql", ".r", ".m", ".lua"
)

MAX_FILE_SIZE_KB = 5000 
MAX_TOTAL_CHARS  = 800000

def call_llm(prompt):
    llm = ChatOpenAI(base_url="https://genailab.tcs.in",model="azure/genailab-maas-gpt-4o",api_key="sk-mh7Mypw94_Xp8iF--2mFng",http_client=httpx.Client(verify=False),temperature=0.2)
    response = llm.invoke(prompt)
    return response.content


def is_code_file(filename):
    return filename.endswith(CODE_EXTENSIONS)

def safe_read_file(filepath):
    try:
        size_kb = os.path.getsize(filepath) / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            return f"FILE: {filepath}\nSkipped — file too large ({size_kb:.0f} KB)\n"
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f"FILE: {filepath}\n" + f.read()
    except Exception as e:
        return f"FILE: {filepath}\nCould not read: {e}\n"

def collect_files_from_dir(directory):
    parts = []
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            if is_code_file(filename):
                filepath = os.path.join(root, filename)
                content = safe_read_file(filepath)
                if content:
                    parts.append(content)
    return "\n\n".join(parts) if parts else "No valid code files found."

def read_uploaded_file(uploaded_file):
    try:
        return f"FILE: {uploaded_file.name}\n" + uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"


def read_zip(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
        except zipfile.BadZipFile as e:
            return f"Bad ZIP file: {e}"
        return collect_files_from_dir(tmpdir)


def parse_github_url(url):
    url = url.strip().rstrip("/").replace(".git", "")
    if "github.com/" not in url:
        raise ValueError("Not a valid GitHub URL")

    after_domain = url.split("github.com/", 1)[1]
    parts = after_domain.split("/")

    if len(parts) < 2:
        raise ValueError("URL must contain at least owner/repo")

    owner  = parts[0]
    repo   = parts[1]
    branch = None
    subpath = None

    if len(parts) > 2:
        if parts[2] in ("tree", "blob") and len(parts) > 3:
            branch  = parts[3]
            subpath = "/".join(parts[4:]) if len(parts) > 4 else None

    return {"owner": owner, "repo": repo, "branch": branch, "subpath": subpath}

def resolve_default_branch(owner, repo, token):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}",headers=headers, timeout=10, verify=False)
        if r.status_code == 200:
            return r.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"


def fetch_github_tree(owner, repo, branch, token) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(url, headers=headers, timeout=20, verify=False)

    if r.status_code == 404:
        raise ValueError(f"Repo not found or branch '{branch}' doesn't exist.")
    if r.status_code == 403:
        raise ValueError("GitHub API rate limit hit. Pass a token in Settings.")
    r.raise_for_status()

    data = r.json()
    if data.get("truncated"):
        st.warning("Repository tree was truncated by GitHub (very large repo). Some files may be missing.")

    return [item for item in data.get("tree", []) if item["type"] == "blob"]


def fetch_blob_content(owner, repo, file_path, branch, token):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    r = requests.get(url, headers=headers, timeout=15, verify=False)

    if r.status_code != 200:
        return None

    data = r.json()
    size_kb = data.get("size", 0) / 1024

    if size_kb > MAX_FILE_SIZE_KB:
        return f"#Skipped {file_path} — too large ({size_kb:.0f} KB)\n"

    encoding = data.get("encoding", "")
    content  = data.get("content", "")

    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return None

    return content or None


def fetch_github_code(url, token):
    try:
        info   = parse_github_url(url)
        owner  = info["owner"]
        repo   = info["repo"]
        branch = info["branch"] or resolve_default_branch(owner, repo, token)
        filter_prefix = info["subpath"] 
        st.info(f"📡 Fetching `{owner}/{repo}` @ branch `{branch}` ")
        tree = fetch_github_tree(owner, repo, branch, token)
        code_files = [
            item for item in tree
            if is_code_file(item["path"])
            and (filter_prefix is None or item["path"].startswith(filter_prefix))
        ]
        if not code_files:
            return "No valid code files found in repository."
        st.info(f"Found {len(code_files)} code files — fetching contents ")
        parts = []
        progress = st.progress(0)
        for i, item in enumerate(code_files):
            content = fetch_blob_content(owner, repo, item["path"], branch, token)
            if content:
                parts.append(f"FILE: {item['path']}\n{content}")
            progress.progress((i + 1) / len(code_files))
        progress.empty()
        if not parts:
            return "All files were empty or unreadable."
        return "\n\n" + "\n\n".join(parts)

    except ValueError as e:
        return f"{e}"
    except Exception as e:
        return f"Unexpected error: {e}"

@st.cache_resource(show_spinner=False)
def build_vector_store(code_text):
    splitter = CodeAwareTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_text(code_text)
    docs = [Document(page_content=c) for c in chunks]
    embeddings = OpenAIEmbeddings(base_url="https://genailab.tcs.in",api_key="sk-mh7Mypw94_Xp8iF--2mFng",model="azure/genailab-maas-text-embedding-3-large ")
    return FAISS.from_documents(docs, embeddings)

def retrieve_relevant_chunks(query, vector_store, k = 8):
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def get_code_for_prompt(code_text, query):
    if len(code_text) <= MAX_TOTAL_CHARS:
        return code_text

    st.warning(f"Code is large ({len(code_text):,} chars). Using RAG — only the most relevant chunks will be sent to the LLM.")
    with st.spinner("Building embeddings index "):
        vs = build_vector_store(code_text)
    return retrieve_relevant_chunks(query, vs)


def fix_code(code, query="fix bugs"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"Fix all bugs and return ONLY the corrected code:\n\n{chunks}")

def generate_tests(code, query="generate tests"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"Please write test cases using python, java or any other relevant test cases like using junit, selenium or any other testing framework \n\n{chunks}")

def enhance_code(code, query="optimize improve"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"Improve and optimize this code for readability, performance, and best practices:\n\n{chunks}")

def get_suggestions(code, query="suggestions improvements"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"Give in Depth actionable improvement suggestions for this code:\n\n{chunks}")

def full_review(code, query="review bugs improvements security"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"""Perform a full code review and report on:
- Bugs & errors
- Security issues
- Performance improvements
- Project specific grammer, conventions, Standard, ethics and architecture
- Code quality & structure
- Suggestions

Code:
{chunks}
""")

def custom(code, query="review bugs improvements security"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"""Explain below points for the code.
{chunks}
""")

def explain(code, query="review bugs improvements security"):
    chunks = get_code_for_prompt(code, query)
    return call_llm(f"""Explain these below points about this project/code like a documentation
- Architechture (draw if necessary)
- Technology
- Languages used
- Components available
- Anything else important or better to know
Code:
{chunks}
""")


st.set_page_config(page_title="AI Code Review Assistant", layout="wide")
st.title("Alpha Code Analyzer")


github_token = "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"

st.subheader("Input Options")
input_mode = st.radio(
    "Choose input type:",
    ["Paste Code", "Upload File", "Upload Folder (ZIP)", "GitHub Link","Custom","Explain"]
)

code_input = ""

if input_mode == "Paste Code":
    code_input = st.text_area("Paste your code here:", height=300)

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload a code file")
    if uploaded_file:
        code_input = read_uploaded_file(uploaded_file)
        st.success("File loaded!")

elif input_mode == "Upload Folder (ZIP)":
    uploaded_zip = st.file_uploader("Upload ZIP folder", type=["zip"])
    if uploaded_zip:
        with st.spinner("Extracting ZIP "):
            code_input = read_zip(uploaded_zip)
        st.success("Folder processed!")

elif input_mode == "GitHub Link":
    github_url = st.text_input("Enter GitHub repo / file URL",)
    if github_url:
        with st.spinner("Fetching code from GitHub "):
            code_input = fetch_github_code(
                github_url,
                token=github_token if github_token else None
            )
        st.success("Code fetched from GitHub Successfully")

elif input_mode == "Custom":
    github_url = st.text_input("Enter GitHub repo / file URL",)
    requirements = st.text_input("Enter your requirements",)
    if github_url:
        with st.spinner("Fetching code from GitHub "):
            code_input = requirements + "\nCode :\n" + fetch_github_code(
                github_url,
                token=github_token if github_token else None
            )
        st.success("Code fetched from GitHub!")

elif input_mode == "Explain":
    github_url = st.text_input("Enter GitHub repo / file URL",)
    if github_url:
        with st.spinner("Fetching code from GitHub "):
            code_input = fetch_github_code(
                github_url,
                token=github_token if github_token else None
            )
        st.success("Code fetched from GitHub!")


if code_input:
    file_count = code_input.count("# FILE:")
    st.caption(f"{len(code_input):,} characters · {file_count} file(s) loaded")
    if len(code_input) > MAX_TOTAL_CHARS:
        st.info(f"Code exceeds {MAX_TOTAL_CHARS:,} chars — RAG will be used automatically.")


st.divider()
action = None
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    if st.button("Fix Code"):      action = "fix"
with col2:
    if st.button("Gen Tests"):     action = "test"
with col3:
    if st.button("Enhance"):       action = "enhance"
with col4:
    if st.button("Suggestions"):   action = "suggest"
with col5:
    if st.button("Full Review"):   action = "review"
with col6:
    if st.button("Custom"):   action = "custom"
with col7:
    if st.button("Explain"):   action = "explain"

if action:
    if not code_input.strip():
        st.warning("Please provide code input first.")
    else:
        with st.spinner("Processing"):
            if action == "fix":      result = fix_code(code_input)
            elif action == "test":   result = generate_tests(code_input)
            elif action == "enhance":result = enhance_code(code_input)
            elif action == "suggest":result = get_suggestions(code_input)
            elif action == "review": result = full_review(code_input)
            elif action == "custom": result = custom(code_input)
            elif action == "explain": result = explain(code_input)
            else:                    result = None

        if result:
            st.subheader("Output")
            st.code(result, language="markdown")