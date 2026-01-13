import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

SCOPES = ["https://mail.google.com/"]

def gmail_authenticate():
    creds_info = dict(st.secrets["gmail_oauth"])
    redirect_uri = "https://phishingdetectiongit-2s7gc97yznq7p9335vr3vh.streamlit.app"

    # Build a web-app flow and set the redirect explicitly
    flow = Flow.from_client_config({"web": creds_info}, scopes=SCOPES)
    flow.redirect_uri = redirect_uri

    # If we're returning from Google, capture the "code" and exchange it
    code = st.query_params.get("code")
    if code:
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
            st.session_state["gmail_creds"] = creds
            return build("gmail", "v1", credentials=creds)
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
            st.stop()

    # Otherwise, start the flow
    auth_url, _ = flow.authorization_url(
      access_type="offline",
      include_granted_scopes="true",
      prompt="consent"
    )
    st.link_button("Authenticate with Google", auth_url)
    st.stop()  # Wait for redirect back with ?code=...

def fetch_messages():
    service = gmail_authenticate()
    profile = service.users().getProfile(userId="me").execute()
    st.write("Authenticated email address:", profile["emailAddress"])
    results = service.users().messages().list(userId="me", q="is:unread", maxResults=20).execute()
    st.write(f"Found {len(results.get('messages', []))} results.")



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
