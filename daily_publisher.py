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

class DailyAgentState(TypedDict):
    messages: List[BaseMessage]
    scheduled_posts: List[dict]
    todays_post: dict
    published_post: dict
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
    """Node 1: Fetch posts scheduled for today from Notion - IMPROVED VERSION"""
    print("📅 STEP 1: Fetching posts scheduled for today...")
    
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        database_id = os.getenv("LINKEDIN_POSTS_DATABASE_ID")
        
        today = datetime.now().date().isoformat()
        current_hour = datetime.now().hour
        
        # More flexible query - get all posts for today that haven't been published
        response = notion.databases.query(
            database_id=database_id,
            filter={
                "and": [
                    {
                        "property": "Post Status",
                        "status": {"equals": "Scheduled"}
                    },
                    {
                        "property": "Scheduled Date",
                        "date": {"equals": today}
                    },
                    {
                        "property": "LinkedIn Post Created", 
                        "checkbox": {"equals": False}
                    },
                    {
                        "property": "Ready for Publishing",
                        "checkbox": {"equals": True}
                    }
                ]
            },
            sorts=[
                {
                    "property": "Posting Priority",
                    "direction": "descending"
                },
                {
                    "property": "Content Quality Score",
                    "direction": "descending"
                },
                {
                    "property": "Scheduled Time",  # Add time-based sorting
                    "direction": "ascending"
                }
            ]
        )
        
        scheduled_posts = []
        for page in response["results"]:
            properties = page["properties"]
            
            # Get scheduled time
            scheduled_time_text = properties.get("Scheduled Time", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "14:00")
            
            # Parse scheduled time
            try:
                scheduled_hour = int(scheduled_time_text.split(":")[0])
            except:
                scheduled_hour = 14  # Default to 2 PM
            
            # IMPROVED LOGIC: More flexible time window
            time_diff = abs(current_hour - scheduled_hour)
            
            # Include posts if:
            # 1. Exact hour match, OR
            # 2. Within 2 hours and high priority, OR  
            # 3. Past due (missed posts), OR
            # 4. No other posts found and it's after 9 AM
            priority = properties.get("Posting Priority", {}).get("select", {}).get("name", "Medium")
            
            should_include = (
                time_diff == 0 or  # Exact time
                (time_diff <= 2 and priority == "High") or  # High priority with wider window
                (current_hour > scheduled_hour and current_hour - scheduled_hour <= 6) or  # Past due but not too old
                (len(scheduled_posts) == 0 and current_hour >= 9)  # Fallback for morning posts
            )
            
            if should_include:
                post_data = {
                    "notion_page_id": page["id"],
                    "title": properties.get("Post Title", {}).get("title", [{}])[0].get("text", {}).get("content", ""),
                    "content": properties.get("Post Content", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                    "content_type": properties.get("Content Type", {}).get("select", {}).get("name", ""),
                    "posting_priority": priority,
                    "quality_score": properties.get("Content Quality Score", {}).get("number", 5),
                    "scheduled_time": scheduled_time_text,
                    "scheduled_date": today,
                    "approval_status": properties.get("Approval Status", {}).get("select", {}).get("name", "Approve"),
                    "time_difference": time_diff,  # Track how far off we are
                    "variations": {
                        "a": properties.get("Variation A - News Commentary", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                        "b": properties.get("Variation B - Personal Experience", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                        "c": properties.get("Variation C - Community Discussion", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
                    }
                }
                scheduled_posts.append(post_data)
        
        print(f"📊 Found {len(scheduled_posts)} posts ready for publishing")
        
        return {
            "scheduled_posts": scheduled_posts,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Found {len(scheduled_posts)} posts ready for publishing today")
            ]
        }
        
    except Exception as e:
        print(f"❌ Failed to fetch scheduled posts: {e}")
        return {
            "scheduled_posts": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error fetching scheduled posts: {str(e)}")
            ]
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
    """Improved post selection with more sophisticated scoring"""
    
    scheduled_posts = state.get("scheduled_posts", [])
    if not scheduled_posts:
        return {"todays_post": {}, "messages": [...]}
    
    current_hour = datetime.now().hour
    
    def calculate_post_score(post):
        """Enhanced scoring algorithm"""
        score = 0
        
        # Priority scoring (unchanged)
        priority = post.get("posting_priority", "Medium")
        if priority == "High":
            score += 15  # Increased weight
        elif priority == "Medium":
            score += 8
        
        # Quality score
        score += post.get("quality_score", 5)
        
        # TIME-BASED SCORING (NEW)
        scheduled_time = post.get("scheduled_time", "14:00")
        try:
            scheduled_hour = int(scheduled_time.split(":")[0])
            time_diff = abs(current_hour - scheduled_hour)
            
            # Perfect timing bonus
            if time_diff == 0:
                score += 5
            elif time_diff == 1:
                score += 3
            elif time_diff <= 2:
                score += 1
        except:
            pass
        
        # CONTENT TYPE SCORING (NEW)
        content_type = post.get("content_type", "")
        type_scores = {
            "github_project": 3,    # Your strength
            "news_article": 2,
            "tutorial": 4,
            "commentary": 2
        }
        score += type_scores.get(content_type, 1)
        
        # ENGAGEMENT POTENTIAL (NEW)
        # Check if post has engagement triggers
        content = post.get("content", "")
        if content.strip().endswith('?'):
            score += 2
        if any(trigger in content.lower() for trigger in ['what do you think', 'share your', 'drop a comment']):
            score += 1
        
        return score
    
    # Sort posts by score
    scored_posts = [(post, calculate_post_score(post)) for post in scheduled_posts]
    scored_posts.sort(key=lambda x: x[1], reverse=True)
    
    best_post = scored_posts[0][0]
    best_score = scored_posts[0][1]
    
    print(f"✅ Selected: {best_post.get('title', 'Unknown')[:50]}...")
    print(f"📊 Score: {best_score} | Priority: {best_post.get('posting_priority')} | Quality: {best_post.get('quality_score')}/10")
    
    return {
        "todays_post": best_post,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Selected best post with score {best_score}")
        ]
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

def publish_to_linkedin(state: DailyAgentState) -> DailyAgentState:
    """Node 3: Actually publish the selected post to LinkedIn with media support"""
    print("🚀 STEP 3: Publishing to LinkedIn...")
    
    todays_post = state.get("todays_post", {})
    
    if not todays_post:
        print("ℹ️ No post selected for publishing")
        return {
            "published_post": {},
            "messages": state.get("messages", []) + [
                SystemMessage(content="No post selected for publishing")
            ]
        }
    
    try:
        import requests
        
        # Get LinkedIn credentials
        access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        user_id = os.getenv('LINKEDIN_USER_ID')
        
        if not access_token or not user_id:
            raise ValueError("Missing LinkedIn credentials: LINKEDIN_ACCESS_TOKEN and LINKEDIN_USER_ID required")
        
        post_content = todays_post.get("content", "")
        
        # Check for images to include
        image_urls = []
        
        # Get images from various sources
        if todays_post.get("image_url"):
            image_urls.append(todays_post["image_url"])
        elif todays_post.get("images", {}).get("primary_image"):
            image_urls.append(todays_post["images"]["primary_image"])
        elif todays_post.get("images", {}).get("readme_images"):
            image_urls.extend(todays_post["images"]["readme_images"][:1])  # Take first image
        
        # Prepare LinkedIn post data
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        
        # Base post structure
        post_data = {
            "author": f"urn:li:person:{user_id}",
            "lifecycleState": "PUBLISHED",
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
        }
        
        # Handle media upload if images exist
        if image_urls:
            print(f"📸 Uploading {len(image_urls)} image(s) to LinkedIn...")
            
            media_assets = []
            for img_url in image_urls[:1]:  # LinkedIn allows max 1 image per post for UGC
                asset_urn = upload_image_to_linkedin(img_url, access_token, user_id)
                if asset_urn:
                    media_assets.append({
                        "status": "READY",
                        "description": {
                            "text": "Project screenshot"
                        },
                        "media": asset_urn,
                        "title": {
                            "text": todays_post.get("title", "Project Update")
                        }
                    })
            
            if media_assets:
                post_data["specificContent"] = {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": post_content},
                        "shareMediaCategory": "IMAGE",
                        "media": media_assets
                    }
                }
                print("✅ Post prepared with media")
            else:
                # Fallback to text-only if image upload failed
                post_data["specificContent"] = {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": post_content},
                        "shareMediaCategory": "NONE"
                    }
                }
                print("⚠️ Image upload failed, posting text-only")
        else:
            # Text-only post
            post_data["specificContent"] = {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": post_content},
                    "shareMediaCategory": "NONE"
                }
            }
            print("📝 Posting text-only content")
        
        # Make the API call
        url = "https://api.linkedin.com/v2/ugcPosts"
        response = requests.post(url, headers=headers, json=post_data)
        
        if response.status_code == 201:
            linkedin_response = response.json()
            linkedin_post_id = linkedin_response.get("id", f"linkedin_post_{int(time.time())}")
            
            # Update Notion with published status
            if todays_post.get("notion_page_id"):
                try:
                    notion = Client(auth=os.getenv("NOTION_TOKEN"))
                    notion.pages.update(
                        page_id=todays_post["notion_page_id"],
                        properties={
                            "Post Status": {"status": {"name": "Published"}},
                            "LinkedIn Post Created": {"checkbox": True},
                            "Published Date": {"date": {"start": datetime.now().date().isoformat()}},
                            "Actual Publish Time": {"rich_text": [{"text": {"content": datetime.now().strftime("%H:%M")}}]},
                            "LinkedIn Post ID": {"rich_text": [{"text": {"content": linkedin_post_id}}]}
                        }
                    )
                    print("✅ Updated Notion with published status")
                except Exception as e:
                    print(f"⚠️ Failed to update Notion: {e}")
            
            published_post_data = {
                **todays_post,
                "linkedin_post_id": linkedin_post_id,
                "actual_publish_time": datetime.now().isoformat(),
                "post_content": post_content,
                "media_included": len(image_urls) > 0,
                "publish_status": "success"
            }
            
            print("✅ Successfully posted to LinkedIn!")
            print(f"📊 Post ID: {linkedin_post_id}")
            print(f"📸 Media included: {'Yes' if image_urls else 'No'}")
            
            return {
                "published_post": published_post_data,
                "messages": state.get("messages", []) + [
                    SystemMessage(content=f"Successfully published to LinkedIn: {linkedin_post_id}")
                ]
            }
        
        else:
            error_msg = f"LinkedIn API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            
            return {
                "published_post": {
                    **todays_post,
                    "publish_status": "failed",
                    "error_message": error_msg
                },
                "messages": state.get("messages", []) + [
                    SystemMessage(content=f"LinkedIn publishing failed: {error_msg}")
                ]
            }
            
    except Exception as e:
        error_msg = f"LinkedIn publishing failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "published_post": {
                **todays_post,
                "publish_status": "failed", 
                "error_message": error_msg
            },
            "messages": state.get("messages", []) + [
                SystemMessage(content=error_msg)
            ]
        }

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
daily_graph.add_node("publish_post", publish_to_linkedin)
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
        "todays_post": {},
        "published_post": {},
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