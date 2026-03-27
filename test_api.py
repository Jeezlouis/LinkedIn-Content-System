import os
import requests
from dotenv import load_dotenv
import tweepy
from notion_client import Client

load_dotenv()

def test_notion():
    print("--- Testing Notion ---")
    token = os.getenv("NOTION_TOKEN")
    if not token:
        print("❌ NOTION_TOKEN missing")
        return
    client = Client(auth=token)
    try:
        # Just check if we can reach Notion
        client.users.me()
        print("✅ Notion API Connection Successful")
    except Exception as e:
        print(f"❌ Notion API Connection Failed: {e}")

    # Check database IDs
    dbs = {
        "NEWS_ARTICLE_DATABASE_ID": os.getenv("NEWS_ARTICLE_DATABASE_ID") or os.getenv("DATABASE_ID"),
        "LINKEDIN_POSTS_DATABASE_ID": os.getenv("LINKEDIN_POSTS_DATABASE_ID"),
        "X_POSTS_DATABASE_ID": os.getenv("X_POSTS_DATABASE_ID"),
        "GITHUB_DATABASE_ID": os.getenv("GITHUB_DATABASE_ID")
    }
    for name, db_id in dbs.items():
        if not db_id:
            print(f"⚠️ {name} is not set in .env")
        else:
            try:
                client.databases.retrieve(database_id=db_id)
                print(f"✅ {name} (ID: {db_id[:5]}...) found")
            except Exception as e:
                print(f"❌ {name} (ID: {db_id[:5]}...) not found or inaccessible: {e}")

def test_linkedin():
    print("\n--- Testing LinkedIn ---")
    token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    user_id = os.getenv("LINKEDIN_USER_ID")
    if not token or not user_id:
        print("❌ LinkedIn credentials missing in .env")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    
    # Try the legacy endpoint first (r_liteprofile)
    try:
        print("🔍 Checking legacy profile endpoint...")
        res = requests.get("https://api.linkedin.com/v2/me", headers=headers)
        if res.status_code == 200:
            profile = res.json()
            print(f"✅ LinkedIn Legacy Connection Successful. User: {profile.get('localizedFirstName')} {profile.get('localizedLastName')}")
            return
        elif res.status_code == 403:
            print("⚠️ Legacy endpoint not available (Expected for newer apps). Checking OpenID Connect...")
            # Try the new OpenID Connect endpoint (openid profile)
            res = requests.get("https://api.linkedin.com/v2/userinfo", headers=headers)
            if res.status_code == 200:
                profile = res.json()
                print(f"✅ LinkedIn OIDC Connection Successful. User: {profile.get('name')}")
                print(f"✅ LinkedIn ID: {profile.get('sub')}")
                print(f"💡 Copy 'sub' if yours doesn't match .env: {user_id}")
                return
            else:
                print(f"❌ LinkedIn OIDC Failed too ({res.status_code}): {res.text}")
        else:
            print(f"❌ LinkedIn Connection Failed ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ LinkedIn error: {e}")
    
    print("\n💡 Tip: Ensure 'Share on LinkedIn' is added to your app Products")
    print("💡 Tip: Generate a NEW token after adding products/permissions")

def test_x():
    print("\n--- Testing X (Twitter) ---")
    try:
        client = tweepy.Client(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_KEY_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
        )
        # Check if we can get authenticated user info
        user = client.get_me()
        if user and user.data:
            print(f"✅ X Connection Successful. User: @{user.data.username}")
        else:
            print("❌ X Connection Failed: No user data returned")
    except Exception as e:
        print(f"❌ X Connection Failed: {e}")

def test_github():
    print("\n--- Testing GitHub ---")
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN missing")
        return
    try:
        from github import Github
        g = Github(token)
        user = g.get_user()
        print(f"✅ GitHub Connection Successful. User: {user.login}")
    except Exception as e:
        print(f"❌ GitHub Connection Failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting API Connection Verification...")
    test_notion()
    test_github()
    test_linkedin()
    test_x()
    print("\nVerification Complete!")
