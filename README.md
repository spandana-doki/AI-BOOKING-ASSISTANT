## AI Booking Assistant

An AI-powered booking assistant built with Streamlit that can:
- Chat with users to create and confirm bookings.
- Answer general questions using PDF-based Retrieval-Augmented Generation (RAG).
- Store confirmed bookings in a SQLite database.
- Provide a read-only admin dashboard for viewing and filtering bookings.

The project is designed with a clear separation between UI, business logic, and data/storage layers.

---

### Features

- **Conversational booking flow**
  - Collects name, email, phone, booking type, date, and time.
  - Validates email, date, and time formats.
  - Summarizes booking details and asks for explicit yes/no confirmation.

- **RAG-based Q&A**
  - Upload PDFs as a knowledge base.
  - Extracts text, chunks content, embeds with OpenAI, and retrieves relevant passages.
  - Uses retrieved context to answer user questions.

- **Persistent bookings**
  - Uses SQLite (no ORM) for storing customers and bookings.
  - Clean separation between booking flow (logic) and persistence (tools/database).

- **Admin dashboard**
  - Read-only view of all bookings.
  - Search/filter by customer name or email.

- **Simple tools layer**
  - RAG tool, booking persistence tool, and email tool with internal error handling.

---

### Tech Stack

- **Frontend/UI**
  - `Streamlit` for the chat interface, PDF upload, and admin dashboard.

- **AI & Retrieval**
  - `google-generativeai` for Gemini text generation and embeddings.
  - `numpy` for vector operations.
  - `faiss-cpu` for efficient vector similarity search.
  - `pypdf` for PDF text extraction.

- **Persistence & Email**
  - Built-in `sqlite3` (standard library) for the database.
  - Built-in `smtplib` and `email` for sending email notifications.

---

### Setup Instructions

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure secrets**

Edit `.streamlit/secrets.toml` and provide real values for:

- `GEMINI_API_KEY`
- `EMAIL_USER`
- `EMAIL_PASSWORD`
- `SMTP_SERVER`
- `SMTP_PORT`

> Do not commit real credentials to version control.

5. **Run the app locally**

```bash
streamlit run app/main.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

### Streamlit Cloud Deployment

1. **Push your code to a Git repository**
   - Ensure `requirements.txt` and `.streamlit/secrets.toml` (with placeholder keys only) are present.

2. **Create a new Streamlit Cloud app**
   - Go to Streamlit Community Cloud.
   - Create a new app and point it to your repo.
   - Set the **main file path** to `app/main.py`.

3. **Configure secrets in Streamlit Cloud**
   - In the app settings, open the **Secrets** section.
   - Paste your secrets in TOML format, e.g.:

```toml
GEMINI_API_KEY = "your_gemini_api_key"

EMAIL_USER = "your_email_username"
EMAIL_PASSWORD = "your_email_password"

SMTP_SERVER = "smtp.yourprovider.com"
SMTP_PORT = "587"
```

4. **Deploy and test**
   - Save the settings; Streamlit Cloud will install dependencies and start the app.
   - Test both the chat booking flow and the admin dashboard from the cloud URL.

---

### Admin Dashboard Usage

- Open the app and use the **sidebar navigation** to switch to **"Admin Dashboard"**.
- The dashboard provides:
  - A **search box** to filter bookings by customer name or email (case-insensitive).
  - A **table view** listing:
    - Booking ID
    - Customer name, email, phone
    - Service/booking type
    - Date and time
    - Status and notes
    - Creation timestamp
- The dashboard is **read-only**:
  - No booking creation, editing, or deletion is performed here.
  - All write operations occur through the chat-based booking flow and tools.

---

### Key Modules Overview

- `app/main.py`: Single entry point for all Streamlit UI (chat, file upload, sidebar navigation, status messages).
- `app/chat_logic.py`: Intent detection, message routing, and history management (no UI code).
- `app/booking_flow.py`: Booking state machine, validation, and confirmation flow (no database access).
- `app/rag_pipeline.py`: PDF ingestion, chunking, embeddings, and retrieval logic (no UI).
- `app/tools.py`: Utility tools (RAG tool, booking persistence tool, email tool) with internal error handling.
- `db/database.py`: Raw SQLite helper functions (connect, init schema, insert/fetch).
- `app/admin_dashboard.py`: Streamlit-based read-only admin dashboard for viewing bookings.


