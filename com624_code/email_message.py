import os
import pickle
# Gmail API utils
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
# for encoding/decoding messages in base64
from base64 import urlsafe_b64decode, urlsafe_b64encode

# Request all access (permission to read/send/receive emails, manage the inbox, and more)
SCOPES = ['https://mail.google.com/']
our_email = 'your_gmail@gmail.com'


def gmail_authenticate():
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)


# get the Gmail API service
service = gmail_authenticate()


def search_messages(service):
    result = service.users().messages().list(userId='me', q="is:unread", maxResults=20).execute()
    messages = []
    if 'messages' in result:
        messages.extend(result['messages'])
    return messages


def read_message(service, message):
    msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
    #print(msg['payload']['parts'][1])
    payload = msg['payload']
    headers = payload.get("headers")
    parts = payload.get("parts")
    header_dict = {h['name']: h['value'] for h in headers}

    # Parse body and attachments
    body = parse_parts(service, parts, message)

    return {'headers': header_dict, 'body': body}


def parse_parts(service, parts, message):
    """
    Utility function that parses the content of an email partition
    """
    if parts:
        for part in parts:
            mimeType = part.get("mimeType")
            body = part.get("body")
            data = body.get("data")
            part_headers = part.get("headers")
            if part.get("parts"):
                # recursively call this function when we see that a part
                # has parts inside
                parse_parts(service, part.get("parts"), message)
            if mimeType == "text/plain":
                # if the email part is text plain
                if data:
                    text = urlsafe_b64decode(data).decode()

                return text

            else:
                # attachment other than a plain text or HTML
                for part_header in part_headers:
                    part_header.get("name")
                    part_header.get("value")


def clean(text):
    # clean text for creating a folder
    return "".join(c if c.isalnum() else "_" for c in text)


def fetch_messages():
    service = gmail_authenticate()
    # Get the profile of the authenticated account
    profile = service.users().getProfile(userId='me').execute()
    print("Authenticated email address:", profile['emailAddress'])
    # get emails that match the query you specify
    results = search_messages(service)
    print(f"Found {len(results)} results.")

    messages_data = []

    for msg in results:
        info = read_message(service, msg)
        messages_data.append(info)

    return messages_data
