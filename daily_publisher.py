from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from notion_client import Client
from datetime import datetime, timezone, timedelta
import os
import json
import re
import time
import schedule
from prompt import performance_analyzer

load_dotenv()

import tweepy # For X integration

def safe_truncate(text, limit):
    """Gracefully truncate text at word boundaries or sentence endings"""
    if len(text) <= limit:
        return text
        
    # Attempt 1: Look for the last sentence ending (. ! ?) before the limit
    for i in range(limit - 3, limit // 2, -1):
        if text[i] in ".!?":
            return text[:i+1]
            
    # Attempt 2: Look for the last space before the limit
    for i in range(limit - 3, limit // 2, -1):
        if text[i] == " ":
            return text[:i].strip() + "..."
            
    # Attempt 3: Hard truncate (fallback)
    return text[:limit-3].strip() + "..."

class DailyAgentState(TypedDict):
    messages: List[BaseMessage]
    scheduled_posts: List[dict]
    todays_posts: List[dict] # Support multi-platform
    published_posts: List[dict]
    performance_data: dict

# Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat")


def predict_engagement_score(predictors):
    """Simple engagement prediction based on post characteristics"""
    score = 5.0  # Base score
    
    # Question posts get higher engagement
    if predictors["has_question"]:
        score += 1.5
    
    # Numbers and stats are engaging
    if predictors["has_numbers"]:
        score += 0.8
    
    # Optimal length (150-300 chars for LinkedIn)
    length = predictors["content_length"]
    if 150 <= length <= 300:
        score += 1.0
    elif 100 <= length <= 500:
        score += 0.5
    
    # Hashtags help discovery
    if predictors["has_hashtags"]:
        score += 0.5
    
    # Optimal posting times (9-11 AM, 1-3 PM, 5-6 PM)
    hour = predictors["posting_time"]
    if hour in [9, 10, 13, 14, 17]:
        score += 1.0
    elif hour in [11, 12, 15, 16, 18]:
        score += 0.5
    
    # Tuesday-Thursday are best days
    day = predictors["posting_day"]
    if day in [1, 2, 3]:  # Tue, Wed, Thu
        score += 0.5
    
    # Quality score influence
    score += (predictors["quality_score"] - 5) * 0.3
    
    # Content type bonuses
    content_type = predictors["content_type"]
    if content_type == "github_project":
        score += 0.7  # Your audience loves your projects
    elif content_type == "tutorial":
        score += 0.5
    
    return min(10, max(1, score))  # Keep between 1-10


# Improved query logic for daily publisher
def fetch_scheduled_posts_for_today(state: DailyAgentState) -> DailyAgentState:
    """Node 1: Fetch posts for all platforms scheduled for today"""
    print("📅 STEP 1: Fetching posts scheduled for today across all platforms...")
    
    notion = Client(auth=os.getenv("NOTION_TOKEN"))
    platforms = [
        {"name": "LinkedIn", "db_id": os.getenv("LINKEDIN_POSTS_DATABASE_ID")},
        {"name": "X", "db_id": os.getenv("X_POSTS_DATABASE_ID")},
        {"name": "Threads", "db_id": os.getenv("THREADS_POSTS_DATABASE_ID")}
    ]
    
    today = datetime.now().date().isoformat()
    current_hour = datetime.now().hour
    all_scheduled = []

    for platform in platforms:
        db_id = platform["db_id"]
        if not db_id: continue
        
        print(f"🔍 Checking {platform['name']} database...")
        try:
            # Query Logic (simplified for multi-platform)
            response = notion.databases.query(
                database_id=db_id,
                filter={
                    "and": [
                        {"property": "Post Status", "status": {"equals": "Scheduled"}},
                        {"property": "Scheduled Date", "date": {"equals": today}},
                        {"property": "Ready for Publishing", "checkbox": {"equals": True}}
                    ]
                }
            )
            
            for page in response["results"]:
                props = page["properties"]
                post = {
                    "platform": platform["name"].lower(),
                    "notion_page_id": page["id"],
                    "title": props.get("Post Title", {}).get("title", [{}])[0].get("text", {}).get("content", ""),
                    "content": props.get("Post Content", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                    "priority": props.get("Posting Priority", {}).get("select", {}).get("name", "Medium"),
                    "quality_score": props.get("Content Quality Score", {}).get("number", 5),
                    "scheduled_time": props.get("Scheduled Time", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "14:00"),
                    "image_url": props.get("Image URL", {}).get("url", "")
                }
                # Handle threads for X and Threads
                if platform["name"] in ["X", "Threads"]:
                    post["is_thread"] = props.get("Is Thread", {}).get("checkbox", False)
                    # Fetch content from variations or similar? No, I'll store list in Notion or state.
                
                all_scheduled.append(post)
                
        except Exception as e:
            print(f"⚠️ Failed to fetch {platform['name']} posts: {e}")

    return {
        "scheduled_posts": all_scheduled,
        "messages": state.get("messages", []) + [SystemMessage(content=f"Fetched {len(all_scheduled)} total posts.")]
    }


def get_backup_post(state: DailyAgentState) -> DailyAgentState:
    """NEW NODE: Get backup evergreen content if no scheduled posts"""
    
    scheduled_posts = state.get("scheduled_posts", [])
    if scheduled_posts:  # We have posts, no backup needed
        return state
    
    print("🔄 No scheduled posts found, looking for backup content...")
    
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        database_id = os.getenv("LINKEDIN_POSTS_DATABASE_ID")
        
        # Look for high-quality evergreen content that can be reposted
        response = notion.databases.query(
            database_id=database_id,
            filter={
                "and": [
                    {
                        "property": "Post Status",
                        "status": {"equals": "Published"}
                    },
                    {
                        "property": "Content Quality Score",
                        "number": {"greater_than_or_equal_to": 8}
                    },
                    {
                        "property": "Repost Eligible",  # You'd need to add this field
                        "checkbox": {"equals": True}
                    }
                ]
            },
            sorts=[
                {"property": "Published Date", "direction": "ascending"}  # Oldest first for reposting
            ]
        )
        
        if response["results"]:
            backup_post = response["results"][0]  # Take the oldest high-quality post
            
            # Extract backup post data
            properties = backup_post["properties"]
            backup_data = {
                "notion_page_id": backup_post["id"],
                "title": f"[REPOST] {properties.get('Post Title', {}).get('title', [{}])[0].get('text', {}).get('content', '')}",
                "content": f"Sharing this again because it's still relevant:\n\n{properties.get('Post Content', {}).get('rich_text', [{}])[0].get('text', {}).get('content', '')}",
                "content_type": "repost",
                "posting_priority": "Low",
                "quality_score": properties.get("Content Quality Score", {}).get("number", 8),
                "is_backup": True
            }
            
            print(f"📦 Found backup post: {backup_data['title'][:50]}...")
            
            return {
                "scheduled_posts": [backup_data],
                "messages": state.get("messages", []) + [
                    SystemMessage(content="Using backup evergreen content for today")
                ]
            }
    
    except Exception as e:
        print(f"❌ Backup post system failed: {e}")
    
    # Ultimate fallback - motivational/educational evergreen content
    fallback_content = {
        "notion_page_id": None,
        "title": "Daily Development Insight",
        "content": """🚀 Daily reminder for developers:

The best code you'll ever write is the code that solves real problems for real people.

Focus on:
✓ Understanding the problem deeply
✓ Building for your users, not your ego  
✓ Writing maintainable code
✓ Learning from every project

What's one lesson you've learned recently that made you a better developer?

#SoftwareDevelopment #CodingLife #TechCommunity""",
        "content_type": "evergreen",
        "posting_priority": "Low", 
        "quality_score": 6,
        "is_fallback": True
    }
    
    print("🎯 Using fallback evergreen content")
    
    return {
        "scheduled_posts": [fallback_content],
        "messages": state.get("messages", []) + [
            SystemMessage(content="Using fallback evergreen content")
        ]
    }

def select_todays_post(state: DailyAgentState) -> DailyAgentState:
    """Node: Select the top scheduled post for EACH platform for today"""
    scheduled_posts = state.get("scheduled_posts", [])
    if not scheduled_posts:
        return {"todays_posts": [], "messages": state.get("messages", []) + [SystemMessage(content="No posts for today")]}
    
    current_hour = datetime.now().hour
    winners = []
    
    # Group by platform
    by_platform = {}
    for post in scheduled_posts:
        p = post.get("platform", "linkedin").lower()
        if p not in by_platform: by_platform[p] = []
        by_platform[p].append(post)

    for platform, posts in by_platform.items():
        # Simple scoring to pick the best one for this platform
        scored = []
        for p in posts:
            score = p.get("quality_score", 5)
            if p.get("priority") == "High": score += 5
            scored.append((p, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]
        winners.append(winner)
        print(f"✅ Selected for {platform.upper()}: {winner.get('title', 'Untitled')[:40]}...")

    return {
        "todays_posts": winners,
        "messages": state.get("messages", []) + [SystemMessage(content=f"Selected {len(winners)} platform-specific winners for today.")]
    }

def select_best_variation(state: DailyAgentState) -> DailyAgentState:
    """NEW NODE: Choose the best post variation based on current context"""
    
    todays_post = state.get("todays_post", {})
    if not todays_post:
        return state
    
    variations = todays_post.get("variations", {})
    current_hour = datetime.now().hour
    current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
    
    # Choose variation based on time/day
    if current_hour <= 10:  # Morning - Professional tone
        chosen_variation = variations.get("b", "")  # Personal Experience
        variation_reason = "Morning audience prefers personal insights"
    elif current_hour >= 17:  # Evening - Community engagement
        chosen_variation = variations.get("c", "")  # Community Discussion
        variation_reason = "Evening is optimal for community discussions"
    elif current_day >= 5:  # Weekend - More casual
        chosen_variation = variations.get("c", "")  # Community Discussion
        variation_reason = "Weekend audience engages more with discussions"
    else:  # Default business hours
        chosen_variation = variations.get("a", "")  # News Commentary
        variation_reason = "Business hours suit professional commentary"
    
    # Fallback to main content if variation is empty
    if not chosen_variation or len(chosen_variation.strip()) < 50:
        chosen_variation = todays_post.get("content", "")
        variation_reason = "Using main post content as fallback"
    
    # Update the post content
    updated_post = todays_post.copy()
    updated_post["content"] = chosen_variation
    updated_post["variation_used"] = variation_reason
    
    print(f"📝 Selected variation: {variation_reason}")
    
    return {
        "todays_post": updated_post,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Selected post variation: {variation_reason}")
        ]
    }


def upload_image_to_linkedin(image_url, access_token, user_id):
    """Upload an image to LinkedIn and return the asset URN"""
    try:
        import requests
        
        # Step 1: Register upload
        register_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
        
        register_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        register_data = {
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner": f"urn:li:person:{user_id}",
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }
                ]
            }
        }
        
        register_response = requests.post(register_url, headers=register_headers, json=register_data)
        
        if register_response.status_code != 200:
            print(f"⚠️ Failed to register image upload: {register_response.text}")
            return None
            
        register_result = register_response.json()
        upload_url = register_result["value"]["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
        asset_urn = register_result["value"]["asset"]
        
        # Step 2: Download image from URL
        image_response = requests.get(image_url, timeout=30)
        if image_response.status_code != 200:
            print(f"⚠️ Failed to download image from {image_url}")
            return None
        
        # Step 3: Upload image binary
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
        }
        
        upload_response = requests.post(
            upload_url, 
            headers=upload_headers, 
            data=image_response.content
        )
        
        if upload_response.status_code == 201:
            print("✅ Image uploaded successfully to LinkedIn")
            return asset_urn
        else:
            print(f"⚠️ Failed to upload image: {upload_response.text}")
            return None
            
    except Exception as e:
        print(f"⚠️ Image upload error: {e}")
        return None

def publish_all_posts(state: DailyAgentState) -> DailyAgentState:
    """Node: Orchestrate publishing to LinkedIn and X for today's winners"""
    import requests
    posts = state.get("todays_posts", [])
    if not posts: return state
    
    published_results = []
    
    for post in posts:
        platform = post.get("platform", "linkedin").lower()
        dry_run = post.get("dry_run", False) or os.getenv("DRY_RUN", "false").lower() == "true"
        
        if platform == "linkedin":
            res = _publish_to_linkedin(post, dry_run=dry_run)
            published_results.append(res)
        elif platform == "x":
            res = _publish_to_x(post, dry_run=dry_run)
            published_results.append(res)
        elif platform == "threads":
            res = _publish_to_threads(post, dry_run=dry_run)
            published_results.append(res)
            
    return {
        "published_posts": published_results,
        "messages": state.get("messages", []) + [SystemMessage(content=f"Attempted publishing to {len(published_results)} platforms.")]
    }

def _publish_to_linkedin(post_data, dry_run=False):
    """Internal: Actual LinkedIn API call"""
    print(f"{'🧪 DRY RUN: ' if dry_run else '🚀 '}Publishing to LinkedIn: {post_data.get('title')}")
    if dry_run:
        print(f"Content: {post_data.get('content', '')[:100]}...")
        return {"platform": "linkedin", "status": "dry_run", "id": "TEST_ID"}
    try:
        import requests
        access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        user_id = os.getenv('LINKEDIN_USER_ID')
        
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json", "X-Restli-Protocol-Version": "2.0.0"}
        
        # IMAGE UPLOAD LOGIC
        image_url = post_data.get("image_url")
        asset_urn = None
        if image_url and image_url.startswith("http"):
            print(f"📸 Attempting to upload image: {image_url[:50]}...")
            asset_urn = upload_image_to_linkedin(image_url, access_token, user_id)
            if asset_urn:
                print(f"✅ Image registered with URN: {asset_urn}")
            else:
                print("⚠️ Image upload failed, proceeding with text-only post")

        # Determine URN type (person or member) based on ID format
        # Numeric IDs are usually members in new apps, Alpha-numeric are usually persons
        urn_type = "member" if user_id.isdigit() else "person"
        
        payload = {
            "author": f"urn:li:{urn_type}:{user_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": post_data.get("content", "")},
                    "shareMediaCategory": "IMAGE" if asset_urn else "NONE",
                    "media": [{
                        "status": "READY",
                        "description": {"text": post_data.get("title", "")},
                        "media": asset_urn,
                        "title": {"text": post_data.get("title", "")}
                    }] if asset_urn else []
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
        }
        
        res = requests.post("https://api.linkedin.com/v2/ugcPosts", headers=headers, json=payload)
        if res.status_code == 201:
            print("✅ LinkedIn success!")
            return {"platform": "linkedin", "status": "success", "id": res.json().get("id")}
        
        print(f"❌ LinkedIn failed ({res.status_code}): {res.text}")
        return {"platform": "linkedin", "status": "failed", "error": res.text}
    except Exception as e: 
        print(f"❌ LinkedIn error: {e}")
        return {"platform": "linkedin", "status": "failed", "error": str(e)}

def _publish_to_x(post_data, dry_run=False):
    """Internal: Actual X API call using Tweepy, supporting threads and character limits"""
    title = post_data.get('title', 'Untitled')
    print(f"{'🧪 DRY RUN: ' if dry_run else '🐦 '}Publishing to X: {title}")
    
    content = post_data.get('content', '')
    is_thread = post_data.get('is_thread', False)
    
    # Platform limit for X free accounts
    X_LIMIT = 280
    
    if dry_run:
        print(f"Content length: {len(content)}")
        print(f"Is Thread: {is_thread}")
        return {"platform": "x", "status": "dry_run", "id": "TEST_ID"}
        
    try:
        import tweepy
        client = tweepy.Client(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_KEY_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
        )
        
        # If it's a thread, we expect content to be split by " --- "
        if is_thread and " --- " in content:
            posts = [p.strip() for p in content.split(" --- ") if p.strip()]
            last_tweet_id = None
            
            for i, tweet_text in enumerate(posts):
                # Graceful truncate for free accounts
                tweet_text = safe_truncate(tweet_text, X_LIMIT)
                
                if i == 0:
                    response = client.create_tweet(text=tweet_text)
                else:
                    response = client.create_tweet(text=tweet_text, in_reply_to_tweet_id=last_tweet_id)
                
                last_tweet_id = response.data['id']
                print(f"  ✅ Tweet {i+1} sent")
            
            print("✅ X Thread success!")
            return {"platform": "x", "status": "success", "id": last_tweet_id}
        else:
            # Single tweet
            content = safe_truncate(content, X_LIMIT)
            response = client.create_tweet(text=content)
            print("✅ X success!")
            return {"platform": "x", "status": "success", "id": response.data['id']}
            
    except Exception as e: 
        print(f"❌ X error: {e}")
        return {"platform": "x", "status": "failed", "error": str(e)}

def _publish_to_threads(post_data, dry_run=False):
    """Internal: Actual Threads API call using Meta Graph API, supporting character limits"""
    title = post_data.get('title', 'Untitled')
    print(f"{'🧪 DRY RUN: ' if dry_run else '🧵 '}Publishing to Threads: {title}")
    
    # Threads character limit
    THREADS_LIMIT = 500
    
    content = safe_truncate(post_data.get("content", ""), THREADS_LIMIT)

    if dry_run:
        print(f"Content length: {len(content)}")
        return {"platform": "threads", "status": "dry_run", "id": "TEST_ID"}
    
    try:
        import requests
        access_token = os.getenv('THREADS_ACCESS_TOKEN')
        user_id = os.getenv('THREADS_USER_ID')
        
        if not access_token or not user_id:
            print("❌ Threads missing credentials (THREADS_ACCESS_TOKEN or THREADS_USER_ID)")
            return {"platform": "threads", "status": "failed", "error": "Missing credentials"}

        # Step 1: Create a Threads Media Container
        url = f"https://graph.threads.net/v1.0/{user_id}/threads"
        
        payload = {
            "media_type": "TEXT",
            "text": content,
            "access_token": access_token
        }
        
        # Add image if available
        image_url = post_data.get("image_url")
        if image_url and image_url.startswith("http"):
            payload["media_type"] = "IMAGE"
            payload["image_url"] = image_url

        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ Threads container creation failed: {response.text}")
            return {"platform": "threads", "status": "failed", "error": response.text}
            
        creation_id = response.json().get("id")
        
        # Step 2: Publish the container
        publish_url = f"https://graph.threads.net/v1.0/{user_id}/threads_publish"
        publish_payload = {
            "creation_id": creation_id,
            "access_token": access_token
        }
        
        publish_response = requests.post(publish_url, data=publish_payload)
        if publish_response.status_code == 200:
            print("✅ Threads success!")
            return {"platform": "threads", "status": "success", "id": publish_response.json().get("id")}
        
        print(f"❌ Threads publish failed: {publish_response.text}")
        return {"platform": "threads", "status": "failed", "error": publish_response.text}
        
    except Exception as e:
        print(f"❌ Threads error: {e}")
        return {"platform": "threads", "status": "failed", "error": str(e)}

def track_performance(state: DailyAgentState) -> DailyAgentState:
    """Enhanced performance tracking with better initial analysis"""
    
    published_post = state.get("published_post", {})
    if not published_post:
        return state
    
    # Immediate post analysis (before metrics come in)
    post_content = published_post.get("post_content", "")
    
    # Analyze post characteristics that predict performance
    performance_predictors = {
        "has_question": post_content.strip().endswith('?'),
        "has_numbers": bool(re.search(r'\d+', post_content)),
        "has_hashtags": '#' in post_content,
        "content_length": len(post_content),
        "emoji_count": len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', post_content)),
        "posting_time": datetime.now().hour,
        "posting_day": datetime.now().weekday(),
        "content_type": published_post.get("content_type", ""),
        "quality_score": published_post.get("quality_score", 5)
    }
    
    # Predict engagement based on characteristics
    predicted_engagement = predict_engagement_score(performance_predictors)
    
    performance_data = {
        "linkedin_post_id": published_post.get("linkedin_post_id"),
        "post_characteristics": performance_predictors,
        "predicted_engagement": predicted_engagement,
        "tracking_started_at": datetime.now().isoformat(),
        "check_schedule": [
            datetime.now() + timedelta(hours=1),
            datetime.now() + timedelta(hours=6),
            datetime.now() + timedelta(hours=24),
            datetime.now() + timedelta(days=3),
            datetime.now() + timedelta(days=7)
        ]
    }
    
    print(f"📈 Predicted engagement score: {predicted_engagement}/10")
    
    return {
        "performance_data": performance_data,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Performance tracking initialized with {predicted_engagement}/10 predicted engagement")
        ]
    }

def analyze_performance_and_learn(state: DailyAgentState) -> DailyAgentState:
    """Node 5: Analyze performance data and extract learning insights"""
    print("🧠 STEP 5: Performance analysis and learning...")
    
    published_post = state.get("published_post", {})
    performance_data = state.get("performance_data", {})
    
    if not published_post or not performance_data:
        print("ℹ️ No performance data to analyze yet")
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No performance data available for analysis")
            ]
        }
    
    try:
        # For now, simulate performance analysis
        # TODO: Replace with actual LinkedIn API metrics after some time has passed
        
        # Simulate some basic metrics for demonstration
        simulated_metrics = {
            "likes": 5,  # Would be real data from LinkedIn API
            "comments": 2,
            "shares": 1,
            "views": 100,
            "clicks": 10,
            "engagement_rate": 0.18  # (likes + comments + shares) / views
        }
        
        post_content = published_post.get("post_content", "")
        content_type = published_post.get("content_type", "")
        posting_priority = published_post.get("posting_priority", "")
        
        # Use your existing performance analyzer prompt
        system_prompt = performance_analyzer(
            post_text=post_content,
            engagement_metrics=simulated_metrics,
            comment_themes=["positive feedback", "questions about implementation"],
            posting_details={
                "content_type": content_type,
                "priority": posting_priority,
                "scheduled_time": published_post.get("scheduled_time", ""),
                "actual_time": published_post.get("actual_publish_time", "")
            },
            comparison_data={"average_likes": 8, "average_comments": 3, "average_shares": 1}
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze this post's performance and provide learning insights.")
        ]
        
        response = llm.invoke(messages)
        
        # Parse the analysis
        import re
        clean_content = re.sub(r"```(?:json)?|```", "", response.content).strip()
        analysis_result = json.loads(clean_content)
        
        # Extract key insights
        success_level = analysis_result.get("performance_summary", {}).get("overall_success_level", "average")
        success_factors = analysis_result.get("success_factors", [])
        content_lessons = analysis_result.get("content_lessons", [])
        
        print(f"📊 Performance Analysis: {success_level.upper()}")
        print(f"✅ Success Factors: {', '.join(success_factors[:2])}")
        print(f"📚 Key Lessons: {', '.join(content_lessons[:2])}")
        
        # Update Notion with analysis
        try:
            notion = Client(auth=os.getenv("NOTION_TOKEN"))
            notion_page_id = published_post.get("notion_page_id")
            
            if notion_page_id:
                notion.pages.update(
                    page_id=notion_page_id,
                    properties={
                        "Performance Analysis": {"rich_text": [{"text": {"content": f"Success Level: {success_level}"}}]},
                        "Success Factors": {"rich_text": [{"text": {"content": "; ".join(success_factors)[:1900]}}]},
                        "Learning Insights": {"rich_text": [{"text": {"content": "; ".join(content_lessons)[:1900]}}]},
                        "Engagement Rate": {"number": simulated_metrics.get("engagement_rate", 0) * 100},
                        "Final Likes": {"number": simulated_metrics.get("likes", 0)},
                        "Final Comments": {"number": simulated_metrics.get("comments", 0)},
                        "Final Shares": {"number": simulated_metrics.get("shares", 0)}
                    }
                )
                print("✅ Updated Notion: Performance analysis saved")
            
        except Exception as e:
            print(f"⚠️ Failed to save analysis to Notion: {e}")
        
        return {
            "performance_data": {
                **performance_data,
                "analysis": analysis_result,
                "metrics": simulated_metrics,
                "analyzed_at": datetime.now().isoformat()
            },
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Performance analysis completed: {success_level} performance with key insights extracted")
            ]
        }
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return {
            "performance_data": performance_data,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Performance analysis failed: {str(e)}")
            ]
        }

# Build the daily publishing graph
daily_graph = StateGraph(DailyAgentState)

# Add nodes
daily_graph.add_node("fetch_scheduled", fetch_scheduled_posts_for_today)
daily_graph.add_node("select_post", select_todays_post)
daily_graph.add_node("publish_post", publish_all_posts)
daily_graph.add_node("track_performance", track_performance)
daily_graph.add_node("analyze_performance", analyze_performance_and_learn)
daily_graph.add_node("get_backup", get_backup_post)
daily_graph.add_node("select_variation", select_best_variation)

# Define the flow
daily_graph.add_edge(START, "fetch_scheduled")
daily_graph.add_edge("fetch_scheduled", "get_backup")  # NEW
daily_graph.add_edge("get_backup", "select_post")
daily_graph.add_edge("select_post", "select_variation")  # NEW
daily_graph.add_edge("select_variation", "publish_post")
daily_graph.add_edge("publish_post", "track_performance")
daily_graph.add_edge("track_performance", "analyze_performance")
daily_graph.add_edge("analyze_performance", END)

# Compile the daily graph
daily_app = daily_graph.compile()

def run_daily_publishing():
    """Run the daily publishing and performance tracking pipeline"""
    print("📅 DAILY PUBLISHING & PERFORMANCE PIPELINE")
    print("⚡ Publishing → Tracking → Learning")
    print("=" * 60)
    
    initial_state = {
        "messages": [],
        "scheduled_posts": [],
        "todays_posts": [],
        "published_posts": [],
        "performance_data": {}
    }
    
    try:
        result = daily_app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ DAILY PUBLISHING COMPLETED!")
        print("=" * 60)
        
        published_post = result.get("published_post", {})
        performance_data = result.get("performance_data", {})
        
        if published_post:
            print(f"🚀 Published: {published_post.get('post_title', 'Unknown')}")
            print(f"⏰ Time: {published_post.get('actual_publish_time', 'Unknown')} (Scheduled: {published_post.get('scheduled_time', 'Unknown')})")
            print(f"📊 Priority: {published_post.get('posting_priority', 'Unknown')}")
            print(f"🎯 Quality Score: {published_post.get('quality_score', 'Unknown')}/10")
            
            if performance_data.get("analysis"):
                analysis = performance_data["analysis"]
                success_level = analysis.get("performance_summary", {}).get("overall_success_level", "Unknown")
                print(f"📈 Performance: {success_level.upper()}")
        else:
            print("ℹ️ No post was published today (none scheduled for this time)")
        
        return result
        
    except Exception as e:
        print(f"❌ DAILY PUBLISHING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# Scheduling functions
def schedule_daily_publishing():
    """Schedule daily publishing at multiple optimal times"""
    
    # Schedule publishing at professional peak times (UTC)
    schedule.every().day.at("09:00").do(run_daily_publishing)  # 9 AM - Morning professionals
    schedule.every().day.at("13:00").do(run_daily_publishing)  # 1 PM - Lunch break engagement
    schedule.every().day.at("17:00").do(run_daily_publishing)  # 5 PM - End of workday
    
    print("📅 Daily publishing scheduled for 9 AM, 1 PM, and 5 PM UTC")
    print("⏰ Each run checks for posts scheduled at that time")
    
    while True:
        schedule.run_pending()
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    # For testing - run immediately
    print("🧪 RUNNING DAILY PIPELINE TEST")
    run_daily_publishing()
    
    # For production - run scheduler
    # schedule_daily_publishing()