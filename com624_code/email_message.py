import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from base64 import urlsafe_b64decode

SCOPES = ['https://mail.google.com/']



def gmail_authenticate():
    creds_info = dict(st.secrets["gmail_oauth"])
    flow = InstalledAppFlow.from_client_config({"web": creds_info}, SCOPES)

    # Step 1: Generate the authorization URL
    auth_url, _ = flow.authorization_url(prompt='consent')

    st.markdown(f"[Click here to authenticate with Google]({auth_url})")

    # Step 2: User pastes the code back
    auth_code = st.text_input("Paste the authorization code here:")

    if auth_code:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        st.session_state["gmail_creds"] = creds
        return build('gmail', 'v1', credentials=creds)

    return None



def search_messages(service):
    result = service.users().messages().list(userId='me', q="is:unread", maxResults=20).execute()
    return result.get('messages', [])


def read_message(service, message):
    msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
    payload = msg['payload']
    headers = payload.get("headers", [])
    parts = payload.get("parts", [])
    header_dict = {h['name']: h['value'] for h in headers}
    body = parse_parts(parts)
    return {'headers': header_dict, 'body': body}


def parse_parts(parts):
    if parts:
        for part in parts:
            mimeType = part.get("mimeType")
            body = part.get("body", {})
            data = body.get("data")
            if part.get("parts"):
                return parse_parts(part.get("parts"))
            if mimeType == "text/plain" and data:
                return urlsafe_b64decode(data).decode()
    return ""


def fetch_messages():
    service = gmail_authenticate()
    if not service:
        return []

    profile = service.users().getProfile(userId='me').execute()
    st.write("Authenticated email address:", profile['emailAddress'])

    results = search_messages(service)
    st.write(f"Found {len(results)} results.")

    messages_data = [read_message(service, msg) for msg in results]
    return messages_data
