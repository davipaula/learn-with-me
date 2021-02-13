import os

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.utils.utils import save_as_json

OUTPUT_PATH = (
    "/Users/dnascimentodepau/Documents/personal/projects/vocable_me/data/processed/results.jsonl"
)

CLIENT_SECRETS_FILE = "/secrets/youtube_client.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


def get_authenticated_service():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_console()

    print(credentials)
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)


def youtube_search(client, **kwargs):
    response = client.search().list(**kwargs).execute()

    return response


def get_video_ids(search_criteria: str, max_results: int):
    video_ids = []
    token = None

    client = get_authenticated_service()
    while len(video_ids) < max_results:
        print(f"Getting file {len(video_ids)} of {max_results}")
        response = youtube_search(
            client,
            q=search_criteria,
            pageToken=token,
            part="id,snippet",
            maxResults=50,
            videoCaption="closedCaption",
            type="video",
            regionCode="US",
            videoLicense="creativeCommon",
            relevanceLanguage="en",
            # videoDuration="long",
        )

        if not response["items"]:
            break

        for item in response["items"]:
            video_ids.append(
                {"title": item["snippet"]["title"], "video_id": item["id"]["videoId"]}
            )

        if token == response["nextPageToken"]:
            break

        token = response["nextPageToken"]

    save_as_json(video_ids, OUTPUT_PATH)


if __name__ == "__main__":
    get_video_ids("tedx", 1000)
