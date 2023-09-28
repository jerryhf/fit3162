import os
import hashlib
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
API_KEY = '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'

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def list_drive_files(service, folder_name):
    """List files in a specific folder on Google Drive."""
    query = f"'{folder_name}' in parents"
    results = service.files().list(q=query, pageSize=1000, 
                                   fields="nextPageToken, files(id, name, mimeType, md5Checksum)").execute()
    items = results.get('files', [])
    return items

def get_folder_id(service, folder_name):
    """Get the folder ID of a specific folder on Google Drive."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
    results = service.files().list(q=query, pageSize=1, fields="files(id)").execute()
    items = results.get('files', [])
    if not items:
        return None
    return items[0]['id']

from googleapiclient.discovery import build

def service_gdrive():
    return build('drive', 'v3', developerKey=API_KEY)


def compare_files(service):
    # Get the ID of the Drive folder
    folder_id = get_folder_id(service, "Visual-Context-Attentional-GAN/ch-sims-videos/FOLDERS")
    if not folder_id:
        print("Drive folder not found.")
        return

    # List files in the Drive folder
    drive_files = list_drive_files(service, folder_id)
    
    # Directory to compare files with
    local_directory = "./ch-sims-videos/FOLDERS"
    
    # Iterate over files in the local directory and compare with Drive files
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            local_file_md5 = calculate_md5(local_file_path)
            
            # Check if local file MD5 is in the list of Drive files
            for drive_file in drive_files:
                if drive_file.get('md5Checksum') == local_file_md5:
                    print(f"Match found: {local_file_path} matches with Drive file: {drive_file.get('name')}")
                else:
                    print(f"No match found for: {local_file_path}")

def main():
    service = service_gdrive()
    compare_files(service)

if __name__ == '__main__':
    main()
