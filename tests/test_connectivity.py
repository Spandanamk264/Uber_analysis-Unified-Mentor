import urllib.request
import time
import sys

def check_app():
    url = "http://localhost:8501"
    print(f"Checking {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                print("✅ Web App is ONLINE and responding (Status 200).")
                # Read a bit of content to ensure it's not an error page
                content = response.read(500)
                print(f"Content preview: {content}")
                return True
            else:
                print(f"❌ Web App returned status: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

if __name__ == "__main__":
    # Retry logic
    for i in range(5):
        if check_app():
            sys.exit(0)
        time.sleep(2)
    sys.exit(1)
